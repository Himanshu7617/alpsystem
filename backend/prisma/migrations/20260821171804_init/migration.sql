-- CreateTable
CREATE TABLE "Session" (
    "session_id" TEXT NOT NULL,
    "topic" TEXT NOT NULL,
    "mastery" DOUBLE PRECISION NOT NULL DEFAULT 0.1,
    "asked_questions" JSONB NOT NULL,
    "correct_answers" INTEGER NOT NULL DEFAULT 0,
    "wrong_answers" INTEGER NOT NULL DEFAULT 0,
    "current_difficulty" TEXT NOT NULL DEFAULT 'easy',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "confidence" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "engagement" DOUBLE PRECISION NOT NULL DEFAULT 0.8,
    "cognitive_load" DOUBLE PRECISION NOT NULL DEFAULT 0.2,
    "fatigue" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "history" JSONB NOT NULL DEFAULT '[]',
    "source" TEXT NOT NULL DEFAULT 'synthetic',

    CONSTRAINT "Session_pkey" PRIMARY KEY ("session_id")
);

-- CreateTable
CREATE TABLE "Question" (
    "question_id" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "id" TEXT NOT NULL,
    "difficulty" TEXT NOT NULL,
    "question_type" TEXT NOT NULL DEFAULT 'mcq',
    "question" TEXT NOT NULL,
    "options" JSONB NOT NULL,
    "correct_answer" TEXT NOT NULL,
    "explanation" TEXT NOT NULL,
    "estimated_time_seconds" INTEGER NOT NULL DEFAULT 30,
    "concepts" JSONB NOT NULL DEFAULT '[]',
    "tags" JSONB NOT NULL DEFAULT '[]',
    "learning_objective" TEXT NOT NULL DEFAULT '',
    "prerequisite_level" INTEGER NOT NULL DEFAULT 1,
    "difficulty_score" INTEGER NOT NULL DEFAULT 1,
    "source_type" TEXT NOT NULL DEFAULT 'generated',
    "asked" BOOLEAN NOT NULL DEFAULT false,

    CONSTRAINT "Question_pkey" PRIMARY KEY ("question_id")
);

-- CreateTable
CREATE TABLE "Learner" (
    "id" TEXT NOT NULL,
    "username" TEXT NOT NULL,
    "email" TEXT,
    "passwordHash" TEXT,
    "role" TEXT NOT NULL DEFAULT 'user',
    "concept_mastery" JSONB NOT NULL DEFAULT '{}',
    "roadmap_state" JSONB NOT NULL DEFAULT '{}',
    "total_questions" INTEGER NOT NULL DEFAULT 0,
    "total_correct" INTEGER NOT NULL DEFAULT 0,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Learner_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "QuestionDraft" (
    "id" TEXT NOT NULL,
    "concept_id" TEXT NOT NULL,
    "subject" TEXT NOT NULL,
    "difficulty" TEXT NOT NULL,
    "question" TEXT NOT NULL,
    "option_a" TEXT NOT NULL,
    "option_b" TEXT NOT NULL,
    "option_c" TEXT NOT NULL,
    "option_d" TEXT NOT NULL,
    "correct" TEXT NOT NULL,
    "explanation" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "reviewerNotes" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "QuestionDraft_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "Learner_username_key" ON "Learner"("username");

-- CreateIndex
CREATE UNIQUE INDEX "Learner_email_key" ON "Learner"("email");

-- AddForeignKey
ALTER TABLE "Question" ADD CONSTRAINT "Question_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "Session"("session_id") ON DELETE CASCADE ON UPDATE CASCADE;
