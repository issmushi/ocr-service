from .doc_detector import DocDetector
from .document_detector import DocumentDetector
from .textfields_detector import TextFieldsDetector
from .words_detector import WordsDetector
from .ocr_cyrillic import OCRCyrillic
from .ocr_latin import OCRLatin
from .doctype_angles_classificator import DocTypeAngles
from .deskewer import DocDeskewer
from .page_registration.geometry import PageGeometry

__all__ = [
    "DocDetector", "DocumentDetector", "TextFieldsDetector", "WordsDetector", "OCRCyrillic",
    "OCRLatin", "DocTypeAngles", "DocDeskewer", "PageGeometry",
]
