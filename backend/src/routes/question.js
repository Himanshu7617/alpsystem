import {Router} from "express";
import dotenv from "dotenv";
import { fetch_questions, get_first_question, get_next_question } from "../controllers/question.controller.js";
dotenv.config();

const router = Router();

router.post('/', fetch_questions);
router.get('/start-session', get_first_question);
router.post('/submit', get_next_question);
router.get('/submit', get_next_question);

export default router;
