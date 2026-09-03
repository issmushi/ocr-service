from fastapi import APIRouter

from app.modules.v1.best_frame.router import router as best_frame_router
from app.modules.v1.crop.router import router as crop_router
from app.modules.v1.full.router import router as full_router
from app.modules.v1.idcard.router import router as idcard_router
from app.modules.v1.passport.router import router as passport_router

router = APIRouter(prefix="/v1/ocr")
router.include_router(passport_router)
router.include_router(idcard_router)
router.include_router(full_router)
router.include_router(crop_router)
router.include_router(best_frame_router)
