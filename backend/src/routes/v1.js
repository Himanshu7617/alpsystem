import { Router } from "express";
import {
    post_consent, post_session, get_next_item, post_events, post_attempt, post_session_end
} from "../controllers/v1.controller.js";

const router = Router();

// No auth gate: this is a pilot instrument with no participant recruitment and
// no personal data beyond a display name. Adding one before Phase 10 would only
// break the loop the acceptance test drives.
router.post("/consent", post_consent);
router.post("/sessions", post_session);
router.post("/sessions/end", post_session_end);
router.get("/next-item", get_next_item);
router.post("/events", post_events);
router.post("/attempts", post_attempt);

export default router;
