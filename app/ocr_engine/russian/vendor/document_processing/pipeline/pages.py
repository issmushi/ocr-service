"""Borders, the page canvas and its rectification: page registration of an
internal passport, the card of a vehicle registration certificate,
template-free page geometry, deskew, and the way back to the photo."""
import cv2
import numpy as np

from ..geometry import Chain, Homography
from ..pipeline_modules.doc_detector.image_transformation import (DOC_MARGIN_FRAC, expand_quad,
                                                                   stitch_pages, stitched_geometry)
from ..pipeline_modules.page_registration.geometry import aspect_for


# A Borders quad overlapping the template-registered page by at least this
# much is the same page, and its edges (the physical page edges) give the
# geometry. The template homography replaces it only when the quad is
# clipped by the image border (the mask ran off the frame) and the two
# disagree strongly - otherwise the quad wins even when they differ, because
# on ordinary photos the template fit of page 3 is the less precise of the
# two (its static print is narrow and periodic; measured on samples/).
QUAD_SAME_PAGE_IOU = 0.40


QUAD_CLIPPED_IOU = 0.80


def _segments_payload(meta_results: dict):
    """Normalise DocDetector's contours into plain ``[[x, y], ...]`` lists.

    Only for the conformance probe (``borders.segments``). The stored contours are
    numpy arrays of shape ``(N, 1, 2)`` -- the layout ``cv2.findContours`` returns --
    which no other language reproduces, so a raw dump could not be compared. The
    reshape is O(number of points), a few hundred per document, and it is what makes
    the stage language-neutral rather than a convenience.

    Returns ``None`` when border detection did not run, which the checker treats as a
    stage that legitimately did not happen.
    """
    detector = (meta_results or {}).get('DocDetector') or {}
    segments = detector.get('segm')
    if not segments:
        return None
    out = []
    for contour in segments:
        if contour is None:
            out.append([])
            continue
        pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
        out.append([[float(x), float(y)] for x, y in pts])
    return out


class PagesStage:
    """Pipeline methods: borders, the page canvas and its rectification: page registration of an
    internal passport, the card of a vehicle registration certificate,
    template-free page geometry, deskew, and the way back to the photo.

    A part of Pipeline, not usable on its own: the methods read the
    models and the results that Pipeline.__init__ and process_img set up.
    """

    def _doc_detector(self, img):
        """
        Detect document borders and fix perspective.

        Args:
            img: Input image

        Returns:
            np.ndarray: Image with fixed perspective
        """
        # Only internal passports are photographed as a two-page spread; all
        # other doc types are single cards/pages, so keep just the largest
        # segment (prevents background blobs from being stitched in). The
        # stitch direction is chosen automatically from the pages' layout.
        doc_type = (self.results.doctype or '')
        is_spread = 'intpassport' in doc_type.lower()
        max_pages = 2 if is_spread else 1
        result = self.doc_detector.predict_transform(img, stack='auto', max_pages=max_pages)
        self.results._meta_results = self.results._meta_results | result

        # img = result[self.doc_detector.model_name]['warped_img']
        # return img


    def _geometry_pages(self, img):
        """Template-free page geometry for any document type (page_geometry=True).

        Runs after _doc_detector (and after _register_pages, which takes
        precedence when it rebuilt the canvas). Every Borders page is
        re-cut from a quad of straight lines fitted to its contour, warped
        the way the Borders path warps it (own side lengths, 1 % cushion),
        then straightened by its own lines and unbent by their local tilt.
        Pages, quads and placements are handed over the way DocDetector does,
        so the per-page deskew and field detection work on these pages. The
        evidence goes to meta_results['PageGeometry'].
        """
        det = self.results._meta_results.get('DocDetector') or {}
        reg_info = self.results._meta_results.get('PageRegistration') or {}
        if any(reg_info.get('sources') or []):
            return                      # the registrar already rebuilt this canvas
        segm = det.get('segm') or []
        if not segm:
            return
        geo = self.page_geometry
        quads, quad_infos = geo.quads(segm, img.shape, aspect_for(self.results.doctype))
        pages, used, straightened, page_geometries = [], [], [], []
        for q, qi in zip(quads, quad_infos):
            # quad warp + straightening in one resampling; a side extrapolated
            # past the frame keeps its corners outside the photo
            page, sinfo, geometry = geo.rectify_with_geometry(
                img, q, keep_outside=qi.get('extrapolated') is not None)
            if page is None:
                continue
            pages.append(page)
            used.append(q)
            straightened.append(sinfo)
            page_geometries.append(geometry)
        if not pages:
            return
        stitched, placements = stitch_pages(pages, used, stack='auto')
        det['pages'] = pages if len(pages) >= 2 else []
        det['page_quads'] = used if len(pages) >= 2 else []
        det['page_placements'] = placements if len(pages) >= 2 else []
        det['page_geometries'] = page_geometries if len(pages) >= 2 else []
        det['geometry'] = stitched_geometry(pages, placements, page_geometries)
        det['warped_img'] = stitched
        self.results._meta_results['DocDetector'] = det
        self.results._meta_results['PageGeometry'] = {
            'quads': [{'quad': q.round(1).tolist(), **qi} for q, qi in zip(quads, quad_infos)],
            'straighten': straightened, 'n_pages': len(pages)}

    def _deskew(self, img):
        """Correct residual tilt of the perspective-fixed canvas and store it
        back so img_with_fixed_perspective returns the deskewed image.

        On a two-page spread each page is deskewed ON ITS OWN and the canvas is
        stitched back from the deskewed pages. Two reasons, both real: the pages
        of an open passport rarely lie at the same angle, and the per-page path
        needs the pages themselves (not the canvas) to hand to the field
        detector. The stitched result is still what every consumer sees.

        Deskew stays on after page registration, on purpose: registration
        aligns the printed BLANK, but the personalisation (names, MRZ, the
        vertical series/number) is printed in a separate pass and can sit a
        degree or two off the blank. The projection-profile deskew follows
        the text lines, which is what the word crops and the OCR need
        (measured: skipping it garbled the MRZ's second line on samples/).
        """
        info = self.results._meta_results.get('PageRegistration')
        if info is not None and any(info.get('sources') or []) and self.page_registrar is not None \
                and self.page_registrar.line_refine:
            # The registration pages were already straightened by their own
            # lines (line_refine, blob-tolerant). The projection-profile deskew
            # is NOT run on top of them: on a small photo (canvas ~500 px) it
            # took the dark cushion around a page for text and rotated the
            # straight page by 6 deg, garbling the MRZ (conformance case
            # 12_CR_INTPASSPORT_2011, measured 2026-09-17). The earlier note
            # that deskew had to stay on dates from before line_refine existed.
            info['deskewed'] = False
            return img
        if info is not None:
            info['deskewed'] = True
        meta = self.results._meta_results.get('DocDetector') or {}
        pages = meta.get('pages') or []
        if len(pages) >= 2:
            desked = [self.deskewer.deskew_with_geometry(p) for p in pages]
            desk_pages = [d[0] for d in desked]
            stitched, placements = stitch_pages(desk_pages, meta.get('page_quads') or [],
                                                stack='auto')
            if stitched is not None:
                # each page's own map followed by its own turn (geometry.py)
                page_geometries = [Chain((g,)).then(d[1]) if g is not None else Chain().then(d[1])
                                   for g, d in zip(meta.get('page_geometries') or [None] * len(pages),
                                                   desked)]
                meta['pages'] = desk_pages
                meta['page_placements'] = placements
                meta['page_geometries'] = page_geometries
                meta['geometry'] = stitched_geometry(desk_pages, placements, page_geometries)
                meta['warped_img'] = stitched
                return stitched
        desk, geometry = self.deskewer.deskew_with_geometry(img)
        if self.results._meta_results.get('DocDetector'):
            det = self.results._meta_results['DocDetector']
            det['warped_img'] = desk
            det['geometry'] = Chain((det.get('geometry'),)).then(geometry) \
                if det.get('geometry') is not None else Chain().then(geometry)
        else:
            # no doc_detector result (fallback path) -> stash under Angle90
            self.results._meta_results.setdefault('Angle90', {})['warped_img'] = desk
            self.results._meta_results['Deskew'] = {'geometry': geometry}
        return desk

    def _canvas(self, geometry):
        """The canvas later stages read is the previous one passed through `geometry`.

        Called wherever process_img takes a new image (see geometry.py); None
        means the image was handed on unchanged.
        """
        chain = self.results._meta_results.get('Geometry') or Chain()
        self.results._meta_results['Geometry'] = chain.then(geometry)

    def _borders_geometry(self):
        """Map from the current border canvas back to the image the border stage received.

        DocDetector keeps it up to date through page registration, page geometry
        and the deskew (its 'geometry' key); without a border result the only
        change is a deskew stashed under 'Deskew'.
        """
        det = self.results._meta_results.get('DocDetector')
        if det:
            return det.get('geometry')
        return (self.results._meta_results.get('Deskew') or {}).get('geometry')
