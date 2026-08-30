import {Router} from "express";


const router = Router();

router.post('/login', (req, res) => {
    const username = req.body?.username?.trim();

    if (!username) {
        return res.status(400).json({ message: "Username is required" });
    }

    res.json({
        message: "login success",
        user: { username }
    });
})


export default router;
