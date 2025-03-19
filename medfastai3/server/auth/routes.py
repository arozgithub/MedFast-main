from fastapi import APIRouter
from .auth import signup, login

router = APIRouter()

# Include the authentication routes
router.add_api_route("/signup", signup, methods=["POST"])
router.add_api_route("/login", login, methods=["POST"])
