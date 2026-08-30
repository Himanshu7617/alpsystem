import { jest } from "@jest/globals";
import request from "supertest";
import app from "../app.js";

const TEST_USER = {
    username: `testuser_${Date.now()}`,
    email: `test_${Date.now()}@test.com`,
    password: "testpass123"
};

let token;

describe("Auth + Roadmap Persistence", () => {
    test("POST /auth/register creates a user and returns a JWT", async () => {
        const res = await request(app)
            .post("/auth/register")
            .send(TEST_USER)
            .expect(201);
        expect(res.body).toHaveProperty("token");
        expect(res.body.username).toBe(TEST_USER.username);
        expect(res.body.role).toBe("user");
        token = res.body.token;
    });

    test("POST /auth/login returns a JWT for valid credentials", async () => {
        const res = await request(app)
            .post("/auth/login")
            .send({ email: TEST_USER.email, password: TEST_USER.password })
            .expect(200);
        expect(res.body).toHaveProperty("token");
    });

    test("GET /roadmap returns 401 without JWT", async () => {
        await request(app)
            .get("/roadmap")
            .expect(401);
    });

    test("POST /roadmap/update succeeds with JWT", async () => {
        const res = await request(app)
            .post("/roadmap/update")
            .set("Authorization", `Bearer ${token}`)
            .send({ username: TEST_USER.username, question_id: "ds-e-1", is_correct: true })
            .expect(200);
        expect(res.body).toHaveProperty("roadmap_version");
    });

    test("GET /roadmap returns roadmap data with JWT", async () => {
        const res = await request(app)
            .get("/roadmap")
            .query({ username: TEST_USER.username })
            .set("Authorization", `Bearer ${token}`)
            .expect(200);
        expect(res.body).toHaveProperty("roadmap_version");
        expect(res.body).toHaveProperty("concepts");
        expect(res.body).toHaveProperty("progress");
    });

    test("GET /question/start-session returns 401 without JWT", async () => {
        await request(app)
            .get("/question/start-session")
            .expect(401);
    });
});
