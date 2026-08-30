import jwt from "jsonwebtoken";
const JWT_SECRET = process.env.JWT_SECRET || "dev-secret-change-me";

export const requireAuth = (req, res, next) => {
    const header = req.headers.authorization;
    if (!header || !header.startsWith("Bearer ")) return res.status(401).json({ error: "missing token" });
    try {
        req.user = jwt.verify(header.slice(7), JWT_SECRET);
        next();
    } catch (e) {
        return res.status(401).json({ error: "invalid token" });
    }
};

export const requireAdmin = (req, res, next) => {
    requireAuth(req, res, () => {
        if (req.user.role !== "admin") return res.status(403).json({ error: "admin required" });
        next();
    });
};
