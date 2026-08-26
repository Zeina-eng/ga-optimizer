import numpy as np
import pandas as pd


def generate_student_data(num_students=1000, noise_level=5, random_state=42):
    """
    Generate a synthetic student performance dataset.

    Parameters:
        num_students (int): Number of students.
        noise_level (float): Standard deviation of noise added to final marks.
        random_state (int): Random seed.

    Returns:
        pandas.DataFrame
    """

    np.random.seed(random_state)

    # -------------------------
    # Informative Features
    # -------------------------
    study_hours = np.random.uniform(5, 35, num_students)
    attendance = np.random.uniform(60, 100, num_students)
    sleep_hours = np.random.uniform(4, 9, num_students)
    tutoring_hours = np.random.uniform(0, 8, num_students)
    parental_support = np.random.randint(1, 6, num_students)

    # -------------------------
    # Hidden "True" Final Marks
    # -------------------------
    final_marks = (
        1.2 * study_hours
        + 0.45 * attendance
        + 2.5 * sleep_hours
        + 1.8 * tutoring_hours
        + 3.0 * parental_support
        + np.random.normal(0, noise_level, num_students)
    )

    final_marks = np.clip(final_marks, 0, 100)

    # -------------------------
    # Collinear Features
    # -------------------------
    mock_exam_1 = final_marks + np.random.normal(0, 3, num_students)
    mock_exam_2 = mock_exam_1 + np.random.normal(0, 2, num_students)
    prev_gpa = final_marks / 10 + np.random.normal(0, 0.3, num_students)

    # -------------------------
    # Noise Features
    # -------------------------
    birth_month = np.random.randint(1, 13, num_students)
    favorite_color_id = np.random.randint(1, 11, num_students)
    shoe_size = np.random.randint(35, 46, num_students)
    socioeconomic_noise_1 = np.random.normal(0, 1, num_students)
    socioeconomic_noise_2 = np.random.normal(0, 1, num_students)

    # -------------------------
    # DataFrame
    # -------------------------
    df = pd.DataFrame({
        "Study_Hours": study_hours.round(2),
        "Attendance_Rate": attendance.round(2),
        "Sleep_Hours": sleep_hours.round(2),
        "Tutoring_Hours": tutoring_hours.round(2),
        "Parental_Support": parental_support,
        "Mock_Exam_1": mock_exam_1.round(2),
        "Mock_Exam_2": mock_exam_2.round(2),
        "Prev_GPA": prev_gpa.round(2),
        "Birth_Month": birth_month,
        "Favorite_Color_ID": favorite_color_id,
        "Shoe_Size": shoe_size,
        "Socioeconomic_Noise_1": socioeconomic_noise_1.round(2),
        "Socioeconomic_Noise_2": socioeconomic_noise_2.round(2),
        "Final_Marks": final_marks.round(2)
    })

    return df


# -------------------------
# Test
# -------------------------
if __name__ == "__main__":
    dataset = generate_student_data(10)

    print(dataset)

    dataset.to_csv("student_marks.csv", index=False)

    print("\nDataset saved as student_marks.csv")