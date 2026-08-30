import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import prisma from "../config/db.js";

const JWT_SECRET = process.env.JWT_SECRET || "dev-secret-change-me";
const JWT_EXPIRY = "7d";
const SALT_ROUNDS = 10;

export const register = async (req, res) => {
    try {
        const { username, email, password } = req.body;
        if (!username || !email || !password) return res.status(400).json({ error: "username, email, password required" });
        const existing = await prisma.learner.findFirst({ where: { OR: [{ username }, { email }] } });
        if (existing) return res.status(409).json({ error: "username or email already taken" });
        const passwordHash = await bcrypt.hash(password, SALT_ROUNDS);
        const learner = await prisma.learner.create({ data: { username, email, passwordHash } });
        const token = jwt.sign({ id: learner.id, username, role: learner.role }, JWT_SECRET, { expiresIn: JWT_EXPIRY });
        res.status(201).json({ token, username, role: learner.role });
    } catch (error) {
        console.error("Registration error:", error);
        res.status(500).json({ error: "Internal server error" });
    }
};

export const login = async (req, res) => {
    try {
        const { email, password } = req.body;
        if (!email || !password) return res.status(400).json({ error: "email and password required" });
        const learner = await prisma.learner.findUnique({ where: { email } });
        if (!learner || !learner.passwordHash) return res.status(401).json({ error: "invalid credentials" });
        const valid = await bcrypt.compare(password, learner.passwordHash);
        if (!valid) return res.status(401).json({ error: "invalid credentials" });
        const token = jwt.sign({ id: learner.id, username: learner.username, role: learner.role }, JWT_SECRET, { expiresIn: JWT_EXPIRY });
        res.json({ token, username: learner.username, role: learner.role });
    } catch (error) {
        console.error("Login error:", error);
        res.status(500).json({ error: "Internal server error" });
    }
};
