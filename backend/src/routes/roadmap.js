import { Router } from "express";
import { getRoadmap, updateRoadmap } from "../controllers/roadmap.controller.js";

const router = Router();

router.get("/", getRoadmap);
router.post("/update", updateRoadmap);

export default router;
