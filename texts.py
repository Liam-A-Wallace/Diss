"""Participant-facing instruction text (kwargs for screens.info_screen)."""

WELCOME = dict(
    title="Welcome",
    body=(
        "You will see a series of faces. Some are photographs "
        "of real people; others are AI-generated."
    ),
    steps=[
        ("Look at the face",
         "A face will appear on its own. Take your time to look at it."),
        ("Make your decision",
         "Press ENTER when you are ready, then click REAL or FAKE."),
        ("Rate your confidence",
         "Click a number from 1 (guessing) to 5 (very sure)."),
    ],
)

PRACTICE = dict(
    title="Practice",
    body=(
        "First, you will complete a practice trial to get used "
        "to the task.\n\nYou will not be told whether your answer "
        "is correct."
    ),
    footer="Press ENTER or SPACE to begin",
)

READY = dict(
    title="Ready?",
    body=(
        "The main task starts now.\n\nAs in the practice, you "
        "will not receive feedback. Respond at your own pace "
        "and go with your first impression."
    ),
    footer="Press ENTER or SPACE to begin",
)

BREAK = dict(
    title="Short break",
    body=(
        "You are halfway through.\n\nTake a moment to rest your eyes. "
        "Sit comfortably, facing the screen, and keep your head in the "
        "same position.\n\nPress R if you would like to recalibrate "
        "the eye tracker."
    ),
    footer="Press ENTER or SPACE to continue",
)

THANKS = dict(
    title="Thank you",
    body=(
        "You have completed the task.\n\nYour participation "
        "is greatly appreciated."
    ),
    footer="Press ENTER or SPACE to finish",
)
