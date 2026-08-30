/*
  Warnings:

  - You are about to drop the `QuestionDraft` table. If the table is not empty, all the data it contains will be lost.

*/
-- AlterTable
ALTER TABLE "Session" ADD COLUMN     "consent_id" TEXT,
ADD COLUMN     "ended_at" TIMESTAMP(3),
ADD COLUMN     "learner_id" TEXT,
ADD COLUMN     "started_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN     "version" INTEGER NOT NULL DEFAULT 0;

-- DropTable
DROP TABLE "QuestionDraft";

-- CreateTable
CREATE TABLE "Consent" (
    "consent_id" TEXT NOT NULL,
    "learner_id" TEXT NOT NULL,
    "correctness" BOOLEAN NOT NULL DEFAULT true,
    "timing" BOOLEAN NOT NULL DEFAULT true,
    "interaction" BOOLEAN NOT NULL DEFAULT true,
    "motor" BOOLEAN NOT NULL DEFAULT true,
    "probes" BOOLEAN NOT NULL DEFAULT true,
    "user_agent" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Consent_pkey" PRIMARY KEY ("consent_id")
);

-- CreateTable
CREATE TABLE "ItemAttempt" (
    "attempt_id" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "item_id" TEXT NOT NULL,
    "question_id" TEXT,
    "seq" INTEGER NOT NULL,
    "correct" BOOLEAN NOT NULL,
    "selected_index" INTEGER,
    "skipped" BOOLEAN NOT NULL DEFAULT false,
    "presented_at" TIMESTAMP(3),
    "submitted_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "response_time_ms" INTEGER,
    "features" JSONB NOT NULL DEFAULT '{}',
    "feature_version" TEXT,

    CONSTRAINT "ItemAttempt_pkey" PRIMARY KEY ("attempt_id")
);

-- CreateTable
CREATE TABLE "InteractionEvent" (
    "event_id" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "learner_id" TEXT,
    "item_id" TEXT,
    "attempt_id" TEXT,
    "type" TEXT NOT NULL,
    "client_ts" TIMESTAMP(3) NOT NULL,
    "server_ts" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "seq" INTEGER NOT NULL,
    "payload" JSONB NOT NULL DEFAULT '{}',

    CONSTRAINT "InteractionEvent_pkey" PRIMARY KEY ("event_id")
);

-- CreateTable
CREATE TABLE "CursorSegment" (
    "segment_id" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "attempt_id" TEXT,
    "item_id" TEXT,
    "auc_toward_nonchosen" DOUBLE PRECISION NOT NULL,
    "max_deviation" DOUBLE PRECISION NOT NULL,
    "x_flips" INTEGER NOT NULL,
    "sample_entropy" DOUBLE PRECISION NOT NULL,
    "velocity_peak" DOUBLE PRECISION NOT NULL,
    "velocity_mean" DOUBLE PRECISION NOT NULL,
    "pause_count" INTEGER NOT NULL,
    "path_ratio" DOUBLE PRECISION NOT NULL,
    "time_to_first_movement_ms" INTEGER NOT NULL,
    "time_to_first_selection_ms" INTEGER NOT NULL,
    "hover_time_ms" JSONB NOT NULL DEFAULT '[]',
    "n_samples" INTEGER NOT NULL,
    "sample_interval_ms" INTEGER NOT NULL DEFAULT 50,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "CursorSegment_pkey" PRIMARY KEY ("segment_id")
);

-- CreateTable
CREATE TABLE "Probe" (
    "probe_id" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "attempt_id" TEXT,
    "item_id" TEXT,
    "kind" TEXT NOT NULL,
    "scale_points" INTEGER NOT NULL,
    "shown_at" TIMESTAMP(3) NOT NULL,
    "answered_at" TIMESTAMP(3),
    "response" INTEGER,
    "skipped" BOOLEAN NOT NULL DEFAULT false,
    "assign_p" DOUBLE PRECISION NOT NULL,
    "assign_draw" DOUBLE PRECISION NOT NULL,

    CONSTRAINT "Probe_pkey" PRIMARY KEY ("probe_id")
);

-- CreateTable
CREATE TABLE "StateEstimate" (
    "estimate_id" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "attempt_id" TEXT,
    "knowledge" DOUBLE PRECISION NOT NULL,
    "knowledge_se" DOUBLE PRECISION,
    "knowledge_validated" BOOLEAN NOT NULL DEFAULT false,
    "engagement" DOUBLE PRECISION NOT NULL,
    "engagement_se" DOUBLE PRECISION,
    "engagement_validated" BOOLEAN NOT NULL DEFAULT false,
    "confidence" DOUBLE PRECISION NOT NULL,
    "confidence_se" DOUBLE PRECISION,
    "confidence_validated" BOOLEAN NOT NULL DEFAULT false,
    "fatigue" DOUBLE PRECISION NOT NULL,
    "fatigue_se" DOUBLE PRECISION,
    "fatigue_validated" BOOLEAN NOT NULL DEFAULT false,
    "estimator" TEXT NOT NULL,
    "estimator_version" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "StateEstimate_pkey" PRIMARY KEY ("estimate_id")
);

-- CreateTable
CREATE TABLE "Decision" (
    "decision_id" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "attempt_id" TEXT,
    "policy_name" TEXT NOT NULL,
    "policy_version" TEXT NOT NULL,
    "action" JSONB NOT NULL,
    "action_space" JSONB NOT NULL,
    "propensity" DOUBLE PRECISION NOT NULL,
    "state_snapshot" JSONB NOT NULL,
    "explanation" JSONB NOT NULL DEFAULT '{}',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Decision_pkey" PRIMARY KEY ("decision_id")
);

-- CreateTable
CREATE TABLE "ExperimentAssignment" (
    "assignment_id" TEXT NOT NULL,
    "learner_id" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "arm" TEXT NOT NULL,
    "policy_name" TEXT NOT NULL,
    "seed" INTEGER NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ExperimentAssignment_pkey" PRIMARY KEY ("assignment_id")
);

-- CreateIndex
CREATE INDEX "Consent_learner_id_idx" ON "Consent"("learner_id");

-- CreateIndex
CREATE INDEX "ItemAttempt_session_id_idx" ON "ItemAttempt"("session_id");

-- CreateIndex
CREATE UNIQUE INDEX "ItemAttempt_session_id_seq_key" ON "ItemAttempt"("session_id", "seq");

-- CreateIndex
CREATE INDEX "InteractionEvent_session_id_seq_idx" ON "InteractionEvent"("session_id", "seq");

-- CreateIndex
CREATE INDEX "InteractionEvent_type_idx" ON "InteractionEvent"("type");

-- CreateIndex
CREATE INDEX "CursorSegment_session_id_idx" ON "CursorSegment"("session_id");

-- CreateIndex
CREATE INDEX "Probe_session_id_idx" ON "Probe"("session_id");

-- CreateIndex
CREATE INDEX "Probe_kind_idx" ON "Probe"("kind");

-- CreateIndex
CREATE INDEX "StateEstimate_session_id_idx" ON "StateEstimate"("session_id");

-- CreateIndex
CREATE INDEX "Decision_session_id_idx" ON "Decision"("session_id");

-- CreateIndex
CREATE UNIQUE INDEX "ExperimentAssignment_session_id_key" ON "ExperimentAssignment"("session_id");

-- CreateIndex
CREATE INDEX "ExperimentAssignment_learner_id_idx" ON "ExperimentAssignment"("learner_id");

-- AddForeignKey
ALTER TABLE "ItemAttempt" ADD CONSTRAINT "ItemAttempt_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "Session"("session_id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "InteractionEvent" ADD CONSTRAINT "InteractionEvent_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "Session"("session_id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "CursorSegment" ADD CONSTRAINT "CursorSegment_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "Session"("session_id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "CursorSegment" ADD CONSTRAINT "CursorSegment_attempt_id_fkey" FOREIGN KEY ("attempt_id") REFERENCES "ItemAttempt"("attempt_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Probe" ADD CONSTRAINT "Probe_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "Session"("session_id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Probe" ADD CONSTRAINT "Probe_attempt_id_fkey" FOREIGN KEY ("attempt_id") REFERENCES "ItemAttempt"("attempt_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "StateEstimate" ADD CONSTRAINT "StateEstimate_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "Session"("session_id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "StateEstimate" ADD CONSTRAINT "StateEstimate_attempt_id_fkey" FOREIGN KEY ("attempt_id") REFERENCES "ItemAttempt"("attempt_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Decision" ADD CONSTRAINT "Decision_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "Session"("session_id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Decision" ADD CONSTRAINT "Decision_attempt_id_fkey" FOREIGN KEY ("attempt_id") REFERENCES "ItemAttempt"("attempt_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ExperimentAssignment" ADD CONSTRAINT "ExperimentAssignment_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "Session"("session_id") ON DELETE CASCADE ON UPDATE CASCADE;
