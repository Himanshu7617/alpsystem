# Informed Consent for Adaptive Learning System Study

**Study Title**: Evaluation of AI-Driven Adaptive Question Selection in Computer Science Education  
**Target Sample Size**: n = 30–50 undergraduate Computer Science students  
**Data Retention Period**: 12 months post-study completion  

---

## 1. Study Purpose

You are invited to participate in a research study evaluating an Adaptive Learning Platform (ALP). The purpose of this study is to measure the effectiveness of AI-driven adaptive question selection algorithms compared to traditional rule-based sequencing in improving student mastery, efficiency, and learning outcomes in introductory Computer Science topics.

---

## 2. What You Will Do

If you choose to participate, you will complete two learning sessions (~25–30 minutes each) using the online web platform:
- You will answer multiple-choice and short-answer Computer Science questions drawn from a 126-item Computer Science Education (CSE) question bank.
- You will receive real-time feedback and explanation hints as you navigate the questions.
- You will complete a brief post-session survey rating your learning experience.

Total estimated time commitment is approximately 60 minutes across two counterbalanced sessions.

---

## 3. Behavioral Telemetry & Information Collected

While using the platform, automated behavioral telemetry is logged to evaluate learning patterns and state transitions.

### Data Fields Logged:
The platform measures exactly nineteen (19) aggregate interaction features:
1. `total_response_time` (elapsed seconds per item)
2. `reading_time` (seconds prior to first interaction)
3. `time_after_last_interaction` (idle seconds before submit)
4. `correct` (answer correctness)
5. `attempts` (submission retry count)
6. `skip` (whether item was skipped)
7. `option_changes` (selection toggle count)
8. `mouse_distance` (total cursor travel in pixels)
9. `mouse_speed` (average cursor movement speed in px/sec)
10. `hover_time` (option element hover duration in seconds)
11. `typing_speed` (calculated WPM during response entry)
12. `backspaces` (backspace key count)
13. `delete_frequency` (delete key count)
14. `pause_duration` (typing pause duration in seconds)
15. `question_number` (item position index)
16. `session_duration` (cumulative session time in seconds)
17. `accuracy_decay` (rolling accuracy decay metric)
18. `tab_switches` (focus/blur or tab switch count)
19. `timeRatio` (computed response time ratio relative to item baseline)

### Privacy Protections:
- **No PII Beyond Account Credentials**: No personally identifiable information is recorded beyond your basic study account username/email.
- **No Raw Keystrokes or Pointer Coordinates**: The system **never** records individual keystroke characters or raw $(X, Y)$ mouse cursor coordinates. Only spatial distance/speed totals and timing aggregates are saved.
- **Self-Hosted Security**: Data is hosted on secure, self-hosted infrastructure PostgreSQL database with access restricted to the research team.
- **Retention**: Data will be retained for **12 months after study completion**, after which it will be permanently deleted.
- **No Commercial Sharing**: Data will never be sold, rented, or shared with third parties.

---

## 4. Risks and Benefits

- **Risks**: Risks associated with participation are minimal and no greater than standard computer usage or completing typical online coursework. You may experience slight mental fatigue typical of completing practice quiz items.
- **Benefits**: You will receive personalized practice and immediate feedback on key Computer Science concepts. Your participation contributes to advancing adaptive educational algorithms for future learners.

---

## 5. Voluntary Participation & Right to Withdraw

Participation in this study is entirely voluntary. You may choose to stop participating at any time during or after a session without any penalty, loss of course credit, or impact on your academic standing.

If you decide to withdraw after completing a session, you may request complete deletion of all your collected telemetry and account data by emailing the research team. All data associated with your account will be permanently purged within **30 days** of your request.

---

## 6. Contact Information

If you have questions about the study, your rights as a participant, or wish to request data deletion, please contact:

- **Principal Investigator**: `[PI Name — Placeholder]`  
- **Email**: `[PI Email — placeholder]`  
- **Institutional Ethics Board / IRB**: `[IRB Contact Info — Placeholder]`  

---

## 7. Participant Consent Declaration

By signing below, you confirm that:
- You have read and understood the information provided in this consent form.
- You are at least 18 years of age.
- You voluntarily agree to participate in this study.

<br/>

**Participant Name (Printed)**: ____________________________________________________  

**Participant Signature**: ____________________________________________________  

**Date**: ________________________  
