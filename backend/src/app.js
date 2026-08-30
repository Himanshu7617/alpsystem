import express, {Router} from "express";
import cors from "cors";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import questionRoutes from "./routes/question.js"
import userRoutes from "./routes/user.js"
import roadmapRoutes from "./routes/roadmap.js"
import authRoutes from "./routes/auth.js"
import v1Routes from "./routes/v1.js"
import { get_dashboard } from "./controllers/dashboard.controller.js"
import { requireAuth } from "./middleware/requireAuth.js"
import prisma from "./config/db.js"

dotenv.config();



const app = express();
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const frontendPath = path.resolve(__dirname, "../../files");



app.use(cors())
app.use(express.json());
app.use(express.static(frontendPath));

app.get('/health', async (req, res) => {
    try {
        await prisma.$queryRaw`SELECT 1`;
        return res.json({ status: "ok", database: "up" });
    } catch (error) {
        return res.status(503).json({ status: "degraded", database: "down", error: error.message });
    }
});

// Read-only researcher view over the research tables. No auth gate, for the
// same reason /v1 has none: this is a local pilot instrument, and the page
// shows no answer key and no personal data beyond a display name.
app.get('/research/dashboard', get_dashboard);

app.use('/auth', authRoutes);
app.use('/v1', v1Routes);
app.use('/user', userRoutes);
app.use('/question', requireAuth, questionRoutes);
app.use('/roadmap', requireAuth, roadmapRoutes);





export default app;
