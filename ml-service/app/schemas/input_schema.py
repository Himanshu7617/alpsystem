from pydantic import BaseModel

class PredictionInput(BaseModel):
    isCorrect: bool = False
    timeTaken: float = 30
    attempts: int = 1
    pastAccuracy: float = .5
    difficulty_score: float = 5
    knowledge_before: float = .5
    fatigue_before: float = 0
    total_response_time: float = 30
    reading_time: float = 5
    time_after_last_interaction: float = 1
    skip: bool = False
    option_changes: float = 0
    mouse_distance: float = 0
    mouse_speed: float = 0
    hover_time: float = 0
    typing_speed: float = 0
    backspaces: float = 0
    delete_frequency: float = 0
    pause_duration: float = 0
    question_number: float = 1
    session_duration: float = 30
    tab_switches: float = 0
