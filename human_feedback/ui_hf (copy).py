import os
import sys

# os.environ['ROOT_PATH'] = '/home/rishihazra/PycharmProjects/AutonomousDriving'
sys.path.append(os.environ['ROOT_PATH'])
# os.environ['LOAD_PATH'] = '/home/rishihazra/Downloads'
import glob
import random

# random.seed(10)
import argparse
import pandas as pd
from itertools import combinations
from PyQt5.QtCore import QUrl, Qt, QSize
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
                             QLabel, QCheckBox, QMessageBox, QGroupBox, QProgressBar)


class VideoSurveyApp(QWidget):
    def __init__(self, dummy_run, gen_id):
        super().__init__()
        self.dummy_run = dummy_run
        self.data = pd.DataFrame(columns=['Video 1', 'Video 2', 'Selected',
                                          'Positive Feedback 1', 'Negative Feedback 1',
                                          'Positive Feedback 2', 'Negative Feedback 2'])
        self.video_selected = None  # To track which video is selected
        self.load_dir = os.path.join(os.environ['ROOT_PATH'], 'human_feedback', 'videos_2x')
        GEN_ID = gen_id

        self.save_dir = os.path.join(os.environ['ROOT_PATH'], 'human_feedback', f'generation_{GEN_ID}')
        all_responses = [os.path.basename(f) for f in glob.glob(f'{self.save_dir}/responses_*.csv')]
        PARTICIPANT_ID = 1
        if len(all_responses) != 0:
            prev_id = max([int(response.split('_')[1].split('.')[0]) for response in all_responses])
            PARTICIPANT_ID = prev_id + 1

        video_paths = glob.glob(f'{self.load_dir}/*_{GEN_ID}_*.mp4')
        self.videos = list(combinations(video_paths, 2))
        random.shuffle(self.videos)
        self.videos = random.choices(self.videos, k=5)  # self.videos[20*(PARTICIPANT_ID-1):20*PARTICIPANT_ID]
        self.response_filename = f'responses_{PARTICIPANT_ID}.csv'
        self.save_dir = os.path.join(os.environ['ROOT_PATH'], 'human_feedback', f'generation_{GEN_ID}')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if self.dummy_run:  # for practice, don't save responses
            self.videos = random.choices(self.videos, k=3)
        # Initialize the UI components
        self.currentVideoIndex = 0
        self.init_ui()

    def init_ui(self):
        if self.dummy_run:
            self.setWindowTitle("Autonomous Driving Feedback Survey: Dummy Run")
        else:
            self.setWindowTitle("Autonomous Driving Feedback Survey")
        self.setGeometry(100, 100, 1200, 700)

        # Define font
        question_font = QFont("Arial", 18)  # Increase font size for questions
        option_font = QFont("Arial", 16)  # Slightly smaller font for options

        # Layouts
        self.vbox = QVBoxLayout()
        option_1_box = QHBoxLayout()
        option_2_box = QHBoxLayout()
        video_controls_layout1 = QHBoxLayout()
        video_controls_layout2 = QHBoxLayout()
        full_video_layout1 = QVBoxLayout()
        full_video_layout2 = QVBoxLayout()
        q1_layout_1 = QVBoxLayout()
        q2_layout_1 = QVBoxLayout()
        q1_layout_2 = QVBoxLayout()
        q2_layout_2 = QVBoxLayout()
        navigate_box = QHBoxLayout()

        self.progressBar = QProgressBar()
        self.progressBar.setMaximum(len(self.videos))
        self.progressBar.setFormat("%v/%m")  # Format as "current/total"
        self.progressBar.setFont(question_font)

        # Video players
        self.video1 = QVideoWidget(self)
        self.video2 = QVideoWidget(self)
        self.video1.setFixedSize(320, 240)
        self.video2.setFixedSize(320, 240)

        # Media players
        self.mediaPlayer1 = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer2 = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer1.setVideoOutput(self.video1)
        self.mediaPlayer2.setVideoOutput(self.video2)

        # Play/Pause/Replay Buttons for Video 1
        play_btn1 = QPushButton()
        play_btn1.setIcon(QIcon(QPixmap(os.path.join(os.environ['ROOT_PATH'],
                                                     'human_feedback', "play_button.png"))))
        play_btn1.setStyleSheet("QPushButton {"
                                "border: none;"
                                "background-color: transparent;"
                                "}")
        play_btn1.setIconSize(QSize(50, 50))
        pause_btn1 = QPushButton()
        pause_btn1.setIcon(QIcon(QPixmap(os.path.join(os.environ['ROOT_PATH'],
                                                      'human_feedback', "pause_button.png"))))
        pause_btn1.setStyleSheet("QPushButton {"
                                 "border: none;"
                                 "background-color: transparent;"
                                 "}")
        pause_btn1.setIconSize(QSize(50, 50))
        replay_btn1 = QPushButton()
        replay_btn1.setIcon(QIcon(QPixmap(os.path.join(os.environ['ROOT_PATH'],
                                                       'human_feedback', "replay_button.png"))))
        replay_btn1.setStyleSheet("QPushButton {"
                                  "border: none;"
                                  "background-color: transparent;"
                                  "}")
        replay_btn1.setIconSize(QSize(50, 50))
        play_btn1.clicked.connect(lambda: self.mediaPlayer1.play())
        pause_btn1.clicked.connect(lambda: self.mediaPlayer1.pause())
        replay_btn1.clicked.connect(lambda: (self.mediaPlayer1.setPosition(0), self.mediaPlayer1.play()))
        video_controls_layout1.addWidget(play_btn1)
        video_controls_layout1.addWidget(pause_btn1)
        video_controls_layout1.addWidget(replay_btn1)

        # Play/Pause/Replay Buttons for Video 2
        play_btn2 = QPushButton()
        play_btn2.setIcon(QIcon(QPixmap(os.path.join(os.environ['ROOT_PATH'],
                                                     'human_feedback', "play_button.png"))))
        play_btn2.setStyleSheet("QPushButton {"
                                "border: none;"
                                "background-color: transparent;"
                                "}")
        play_btn2.setIconSize(QSize(50, 50))
        pause_btn2 = QPushButton()
        pause_btn2.setIcon(QIcon(QPixmap(os.path.join(os.environ['ROOT_PATH'],
                                                      'human_feedback', "pause_button.png"))))
        pause_btn2.setStyleSheet("QPushButton {"
                                 "border: none;"
                                 "background-color: transparent;"
                                 "}")
        pause_btn2.setIconSize(QSize(50, 50))
        replay_btn2 = QPushButton()
        replay_btn2.setIcon(QIcon(QPixmap(os.path.join(os.environ['ROOT_PATH'],
                                                       'human_feedback', "replay_button.png"))))
        replay_btn2.setStyleSheet("QPushButton {"
                                  "border: none;"
                                  "background-color: transparent;"
                                  "}")
        replay_btn2.setIconSize(QSize(50, 50))
        play_btn2.clicked.connect(lambda: self.mediaPlayer2.play())
        pause_btn2.clicked.connect(lambda: self.mediaPlayer2.pause())
        replay_btn2.clicked.connect(lambda: (self.mediaPlayer2.setPosition(0), self.mediaPlayer2.play()))
        video_controls_layout2.addWidget(play_btn2)
        video_controls_layout2.addWidget(pause_btn2)
        video_controls_layout2.addWidget(replay_btn2)

        # Buttons for selecting videos
        stylesheet = """
        QPushButton {
            font-size: 30px;
            border: 2px solid #8f8f91;
            border-radius: 10px;
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(255, 255, 255, 255), stop:1 rgba(224, 224, 224, 255));
            padding: 5px;
        }
        QPushButton:hover {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(255, 255, 255, 255), stop:1 rgba(200, 200, 200, 255));
        }
        QPushButton:pressed {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(224, 224, 224, 255), stop:1 rgba(150, 150, 150, 255));
            border-style: inset;
        }
        """
        self.btn_video1 = QPushButton()
        self.btn_video1.setIcon(QIcon(QPixmap(os.path.join(os.environ['ROOT_PATH'],
                                                           'human_feedback', "choose_button.png"))))
        self.btn_video1.setStyleSheet("QPushButton {"
                                      "border: none;"
                                      "background-color: transparent;"
                                      "}")
        self.btn_video1.setIconSize(QSize(100, 100))
        # Set tooltip text to display when hovering
        self.btn_video1.setToolTip("Select Video 1")
        self.btn_video2 = QPushButton()
        self.btn_video2.setIcon(QIcon(QPixmap(os.path.join(os.environ['ROOT_PATH'],
                                                           'human_feedback', "choose_button.png"))))
        self.btn_video2.setStyleSheet("QPushButton {"
                                      "border: none;"
                                      "background-color: transparent;"
                                      "}")
        self.btn_video2.setIconSize(QSize(100, 100))
        self.btn_video2.setToolTip("Select Video 2")
        # self.btn_video1.setMinimumSize(150, 80)
        # self.btn_video1.setMaximumSize(150, 80)
        # self.btn_video2.setMinimumSize(150, 80)
        # self.btn_video2.setMaximumSize(150, 80)
        # self.btn_video1.setStyleSheet(stylesheet)
        # self.btn_video2.setStyleSheet(stylesheet)

        # Next button to process and save responses
        self.btn_next = QPushButton()
        # self.btn_next.setFont(question_font)
        self.btn_next.setIcon(QIcon(QPixmap(os.path.join(os.environ['ROOT_PATH'],
                                                         'human_feedback', "next_button.png"))))
        # Set the icon size\
        self.btn_next.setIconSize(QSize(150, 80))
        # self.btn_next.setGeometry(50, 50, 1000, 50)  # Position and size
        self.btn_next.setStyleSheet("QPushButton {"
                                    "border: none;"
                                    "background-color: transparent;"
                                    "}")

        # Previous button to process and save responses
        self.btn_prev = QPushButton()
        self.btn_prev.setIcon(QIcon(QPixmap(os.path.join(os.environ['ROOT_PATH'],
                                                         'human_feedback', "previous_button.png"))))
        self.btn_prev.setIconSize(QSize(150, 80))
        # self.btn_prev.setGeometry(50, 50, 150, 50)  # Position and size
        self.btn_prev.setStyleSheet("QPushButton {"
                                    "border: none;"
                                    "background-color: transparent;"
                                    "}")

        # Neutral button when both videos are similar and participant cannot decide
        self.neutral_btn = QPushButton("It's a Tie")
        self.neutral_btn.setIcon(QIcon(QPixmap(os.path.join(os.environ['ROOT_PATH'],
                                                            'human_feedback', "tie.png"))))
        self.neutral_btn.setIconSize(QSize(120, 120))
        # self.btn_prev.setGeometry(50, 50, 150, 50)  # Position and size
        self.neutral_btn.setStyleSheet("QPushButton {"
                                       "border: none;"
                                       "background-color: transparent;"
                                       "}")
        self.neutral_btn.setStyleSheet(stylesheet)
        self.neutral_btn.setFont(question_font)

        # Add widgets to layouts
        full_video_layout1.addWidget(self.video1)
        full_video_layout1.addLayout(video_controls_layout1)
        option_1_box.addLayout(full_video_layout1)
        option_1_box.addWidget(self.btn_video1)
        option_1_box.addSpacing(100)  # Adds 100 pixels of spacing
        full_video_layout2.addWidget(self.video2)
        full_video_layout2.addLayout(video_controls_layout2)
        option_2_box.addLayout(full_video_layout2)
        option_2_box.addWidget(self.btn_video2)
        option_2_box.addSpacing(100)  # Adds 100 pixels of spacing

        # Define questions and options
        questions = [
            "What aspects are satisfactory?",
            "What aspects need improvement?"
        ]
        options = [
        "balance stability",
        "forward locomotion quality",
        "smoothness of movement",
        "energy efficiency",
        "recovery capability"
    ]
        # Humanoid locomotion feedback dimensions:
    # - balance stability: ability to stay upright without wobbling or falling
    # - forward locomotion quality: effective, consistent forward walking progress
    # - smoothness of movement: natural, continuous joint motions without jerks
    # - energy efficiency: minimal unnecessary limb flailing or wasted motion
    # - recovery capability: ability to regain stability after small disturbances


        # Add questions and combo boxes for video 1
        group_stylesheet = """
                    QGroupBox {
                        border: 2px solid gray;
                        border-radius: 10px;
                        margin-top: 10px;
                        background-color: #F0F0F0;
                    }
                    QGroupBox:title {
                        subcontrol-origin: margin;
                        subcontrol-position: top center;
                        padding: 5px 10px;
                        background-color: #D0D0D0;
                        color: black;
                        font-weight: bold;
                        font-size: 14px;
                    }
                """
        group_box1_1 = QGroupBox("Video 1")
        group_box1_1.setFont(question_font)  # Set font for the label
        group_box1_1.setStyleSheet(group_stylesheet)  # Custom stylesheet for the group box
        label = QLabel(questions[0])
        label.setFont(question_font)  # Set font for the label
        q1_layout_1.addWidget(label)
        self.checkboxes1_1 = []
        for i, option in enumerate(options):
            chk = QCheckBox(option)
            chk.setFont(option_font)
            self.checkboxes1_1.append(chk)
            q1_layout_1.addWidget(chk)
            chk.stateChanged.connect(lambda state, idx=i: self.sync_checkboxes(state, idx, True, True))
        group_box1_1.setLayout(q1_layout_1)  # Set the layout with widgets to the group box

        # Add questions and combo boxes for video 1
        group_box2_1 = QGroupBox("Video 1")
        group_box2_1.setFont(question_font)  # Set font for the label
        group_box2_1.setStyleSheet(group_stylesheet)  # Custom stylesheet for the group box
        label = QLabel(questions[1])
        label.setFont(question_font)  # Set font for the label
        q2_layout_1.addWidget(label)
        self.checkboxes2_1 = []
        for i, option in enumerate(options):
            chk = QCheckBox(option)
            chk.setFont(option_font)
            self.checkboxes2_1.append(chk)
            q2_layout_1.addWidget(chk)
            chk.stateChanged.connect(lambda state, idx=i: self.sync_checkboxes(state, idx, True, False))
        group_box2_1.setLayout(q2_layout_1)  # Set the layout with widgets to the group box

        # Add questions and combo boxes for video 2
        group_box1_2 = QGroupBox("Video 2")
        group_box1_2.setFont(question_font)  # Set font for the label
        group_box1_2.setStyleSheet(group_stylesheet)  # Custom stylesheet for the group box
        label = QLabel(questions[0])
        label.setFont(question_font)  # Set font for the label
        q1_layout_2.addWidget(label)
        self.checkboxes1_2 = []
        for i, option in enumerate(options):
            chk = QCheckBox(option)
            chk.setFont(option_font)
            self.checkboxes1_2.append(chk)
            q1_layout_2.addWidget(chk)
            chk.stateChanged.connect(lambda state, idx=i: self.sync_checkboxes(state, idx, False, True))
        group_box1_2.setLayout(q1_layout_2)  # Set the layout with widgets to the group box

        group_box2_2 = QGroupBox("Video 2")
        group_box2_2.setFont(question_font)  # Set font for the label
        group_box2_2.setStyleSheet(group_stylesheet)  # Custom stylesheet for the group box
        label = QLabel(questions[1])
        label.setFont(question_font)  # Set font for the label
        q2_layout_2.addWidget(label)
        self.checkboxes2_2 = []
        for i, option in enumerate(options):
            chk = QCheckBox(option)
            chk.setFont(option_font)
            self.checkboxes2_2.append(chk)
            q2_layout_2.addWidget(chk)
            chk.stateChanged.connect(lambda state, idx=i: self.sync_checkboxes(state, idx, False, False))
        group_box2_2.setLayout(q2_layout_2)  # Set the layout with widgets to the group box

        option_1_box.addWidget(group_box1_1)
        option_1_box.addWidget(group_box2_1)
        option_2_box.addWidget(group_box1_2)
        option_2_box.addWidget(group_box2_2)
        self.vbox.addLayout(option_1_box)
        self.vbox.addLayout(option_2_box)

        self.btn_video1.clicked.connect(lambda: self.set_video_selected('1'))
        self.btn_video2.clicked.connect(lambda: self.set_video_selected('2'))
        self.neutral_btn.clicked.connect(lambda: self.set_video_selected('3'))
        self.vbox.addSpacing(50)
        self.btn_next.clicked.connect(lambda: self.record_and_load_next_pair())
        self.btn_prev.clicked.connect(lambda: self.load_prev_pair())
        # self.vbox.addSpacing(50)
        self.vbox.addWidget(self.neutral_btn)

        navigate_box.addWidget(self.btn_prev)
        navigate_box.addWidget(self.progressBar)
        navigate_box.addWidget(self.btn_next)
        self.vbox.addLayout(navigate_box)
        # self.vbox.addWidget(self.progressBar)
        self.setLayout(self.vbox)

        self.start_video()

    def updateProgressBar(self):
        self.progressBar.setValue(self.currentVideoIndex)

    def start_video(self):
        video_file_1, video_file_2 = self.videos[self.currentVideoIndex]
        # Load next videos
        self.mediaPlayer1.setMedia(QMediaContent(QUrl.fromLocalFile(video_file_1)))
        self.mediaPlayer2.setMedia(QMediaContent(QUrl.fromLocalFile(video_file_2)))
        self.mediaPlayer1.play()
        self.mediaPlayer2.play()
        self.updateProgressBar()

    def sync_checkboxes(self, state, index, is_q1, is_first_group):
        if state == Qt.Checked:
            if is_q1:
                if is_first_group:
                    self.checkboxes2_1[index].setChecked(False)
                    self.checkboxes2_1[index].setEnabled(False)
                else:
                    self.checkboxes1_1[index].setChecked(False)
                    self.checkboxes1_1[index].setEnabled(False)
            else:
                if is_first_group:
                    self.checkboxes2_2[index].setChecked(False)
                    self.checkboxes2_2[index].setEnabled(False)
                else:
                    self.checkboxes1_2[index].setChecked(False)
                    self.checkboxes1_2[index].setEnabled(False)
        else:
            if is_q1:
                if is_first_group:
                    self.checkboxes2_1[index].setEnabled(True)
                else:
                    self.checkboxes1_1[index].setEnabled(True)
            else:
                if is_first_group:
                    self.checkboxes2_2[index].setEnabled(True)
                else:
                    self.checkboxes1_2[index].setEnabled(True)

    def load_nxt_and_reset(self):
        self.currentVideoIndex += 1
        # video_file_1, video_file_2 = self.videos[self.currentVideoIndex]
        # Load next videos
        self.start_video()
        self.btn_video1.setIcon(QIcon(QPixmap(os.path.join(os.environ['ROOT_PATH'],
                                                           'human_feedback', "choose_button.png"))))
        self.btn_video1.setStyleSheet("QPushButton {"
                                      "border: none;"
                                      "background-color: transparent;"
                                      "}")
        self.btn_video1.setIconSize(QSize(100, 100))
        self.btn_video2.setIcon(QIcon(QPixmap(os.path.join(os.environ['ROOT_PATH'],
                                                           'human_feedback', "choose_button.png"))))
        self.btn_video2.setStyleSheet("QPushButton {"
                                      "border: none;"
                                      "background-color: transparent;"
                                      "}")
        self.btn_video2.setIconSize(QSize(100, 100))
        # self.btn_video1.setStyleSheet("QPushButton {font-size: 20px; }")
        # self.btn_video2.setStyleSheet("QPushButton {font-size: 20px; }")
        # reset all checkboxes
        for chk11, chk21, chk12, chk22 in (
                zip(self.checkboxes1_1, self.checkboxes2_1, self.checkboxes1_2, self.checkboxes2_2)):
            chk11.setChecked(False)
            chk21.setChecked(False)
            chk12.setChecked(False)
            chk22.setChecked(False)

    def record_and_load_next_pair(self):
        # record responses for current pair, and load next pair
        filepath_1, filepath_2 = self.videos[self.currentVideoIndex]
        filename1 = filepath_1.split('/')[-1].split('.mp4')[0]
        filename2 = filepath_2.split('/')[-1].split('.mp4')[0]

        def checked_option_boxes_1():
            checkboxes1_1_all_checked = all(chk.isChecked() for chk in self.checkboxes1_1)
            checkboxes1_1_none_checked = not any(chk.isChecked() for chk in self.checkboxes1_1)
            checkboxes2_1_all_checked = all(chk.isChecked() for chk in self.checkboxes2_1)
            checkboxes2_1_none_checked = not any(chk.isChecked() for chk in self.checkboxes2_1)

            if (checkboxes1_1_none_checked and checkboxes2_1_all_checked) or (
                    checkboxes1_1_all_checked and checkboxes2_1_none_checked) or (
                    any(chk.isChecked() for chk in self.checkboxes1_1) and
                    any(chk.isChecked() for chk in self.checkboxes2_1)):
                return True
            return False

        def checked_option_boxes_2():
            checkboxes1_2_all_checked = all(chk.isChecked() for chk in self.checkboxes1_2)
            checkboxes1_2_none_checked = not any(chk.isChecked() for chk in self.checkboxes1_2)
            checkboxes2_2_all_checked = all(chk.isChecked() for chk in self.checkboxes2_2)
            checkboxes2_2_none_checked = not any(chk.isChecked() for chk in self.checkboxes2_2)

            if (checkboxes1_2_none_checked and checkboxes2_2_all_checked) or (
                    checkboxes1_2_all_checked and checkboxes2_2_none_checked) or (
                    any(chk.isChecked() for chk in self.checkboxes1_2) and
                    any(chk.isChecked() for chk in self.checkboxes2_2)):
                return True
            return False

        if self.video_selected == '3':
            if (not checked_option_boxes_1()
                    or not checked_option_boxes_2()):
                QMessageBox.warning(self, "Incomplete Response",
                                    "Please answer all feedback questions before proceeding.")
            else:
                self.record_responses(filename1, filename2)
                if self.currentVideoIndex < len(self.videos) - 1:
                    self.load_nxt_and_reset()
                else:
                    QMessageBox.information(self, "End of Survey", "Thank you for participating.")
                    # Save the DataFrame to CSV and show confirmation
                    if not self.dummy_run:
                        self.data.to_csv(os.path.join(self.save_dir, self.response_filename), index=False)
                    self.close()
            return

        # if self.video_selected in ['1', '2']:
        if (self.video_selected is None
                or not checked_option_boxes_1()
                or not checked_option_boxes_2()):
            QMessageBox.warning(self, "Incomplete Response",
                                "Please select a video and answer all feedback questions before proceeding.")
        else:
            self.record_responses(filename1, filename2)
            if self.currentVideoIndex < len(self.videos) - 1:
                self.load_nxt_and_reset()
            else:
                QMessageBox.information(self, "End of Survey", "Thank you for participating.")
                # Save the DataFrame to CSV and show confirmation
                if not self.dummy_run:
                    self.data.to_csv(os.path.join(self.save_dir, self.response_filename), index=False)
                self.close()

    def load_prev_pair(self):
        self.currentVideoIndex -= 1
        if self.currentVideoIndex >= 0:
            self.start_video()
            self.btn_video1.setStyleSheet("QPushButton {font-size: 20px; }")
            self.btn_video2.setStyleSheet("QPushButton {font-size: 20px; }")
            for chk11, chk21, chk12, chk22 in (
                    zip(self.checkboxes1_1, self.checkboxes2_1, self.checkboxes1_2, self.checkboxes2_2)):
                chk11.setChecked(False)
                chk21.setChecked(False)
                chk12.setChecked(False)
                chk22.setChecked(False)
            self.data = self.data[:-1]
        else:
            QMessageBox.information(self, "Invalid Response", "You're on the first pair.")

    def record_responses(self, filename1: str, filename2: str):
        # Record responses for the selected video
        feedback1_1 = [chk.text() for chk in self.checkboxes1_1 if chk.isChecked()]
        feedback2_1 = [chk.text() for chk in self.checkboxes2_1 if chk.isChecked()]
        feedback11_str = ", ".join(feedback1_1)
        feedback21_str = ", ".join(feedback2_1)

        # Record responses for the non-selected video
        feedback1_2 = [chk.text() for chk in self.checkboxes1_2 if chk.isChecked()]
        feedback2_2 = [chk.text() for chk in self.checkboxes2_2 if chk.isChecked()]
        feedback12_str = ", ".join(feedback1_2)
        feedback22_str = ", ".join(feedback2_2)

        if self.video_selected == '3':
            which_video = '0.5'
        else:
            which_video = self.video_selected

        new_row = {'Video 1': filename1, 'Video 2': filename2, 'Selected': which_video,
                   'Positive Feedback 1': feedback11_str, 'Negative Feedback 1': feedback21_str,
                   'Positive Feedback 2': feedback12_str, 'Negative Feedback 2': feedback22_str}
        self.data = pd.concat([self.data, pd.DataFrame([new_row])], ignore_index=True)
        # self.data = self.data.append(new_row, ignore_index=True)

    def set_video_selected(self, video_id):
        if video_id == '1':
            self.btn_video1.setIcon(QIcon(QPixmap(os.path.join(os.environ['ROOT_PATH'],
                                                               'human_feedback', "like.png"))))
            self.btn_video1.setStyleSheet("QPushButton {"
                                          "border: none;"
                                          "background-color: transparent;"
                                          "}")
            self.btn_video1.setIconSize(QSize(100, 100))
            self.btn_video2.setIcon(QIcon(QPixmap(os.path.join(os.environ['ROOT_PATH'],
                                                               'human_feedback', "unlike.png"))))
            self.btn_video2.setStyleSheet("QPushButton {"
                                          "border: none;"
                                          "background-color: transparent;"
                                          "}")
            self.btn_video2.setIconSize(QSize(100, 100))
            # self.btn_video1.setStyleSheet("QPushButton {font-size: 20px; "
            #                               "background-color: green; color: white;}")
            # self.btn_video2.setStyleSheet("QPushButton {font-size: 20px; "
            #                               "background-color: red; color: white;}")
        elif video_id == '2':
            self.btn_video1.setIcon(QIcon(QPixmap(os.path.join(os.environ['ROOT_PATH'],
                                                               'human_feedback', "unlike.png"))))
            self.btn_video1.setStyleSheet("QPushButton {"
                                          "border: none;"
                                          "background-color: transparent;"
                                          "}")
            self.btn_video1.setIconSize(QSize(100, 100))
            self.btn_video2.setIcon(QIcon(QPixmap(os.path.join(os.environ['ROOT_PATH'],
                                                               'human_feedback', "like.png"))))
            self.btn_video2.setStyleSheet("QPushButton {"
                                          "border: none;"
                                          "background-color: transparent;"
                                          "}")
            self.btn_video2.setIconSize(QSize(100, 100))
        else:
            self.btn_video1.setIcon(QIcon(QPixmap(os.path.join(os.environ['ROOT_PATH'],
                                                               'human_feedback', "balanced.png"))))
            self.btn_video1.setStyleSheet("QPushButton {"
                                          "border: none;"
                                          "background-color: transparent;"
                                          "}")
            self.btn_video1.setIconSize(QSize(100, 100))
            self.btn_video2.setIcon(QIcon(QPixmap(os.path.join(os.environ['ROOT_PATH'],
                                                               'human_feedback', "balanced.png"))))
            self.btn_video2.setStyleSheet("QPushButton {"
                                          "border: none;"
                                          "background-color: transparent;"
                                          "}")
        self.video_selected = video_id

    def play_videos(self):
        self.mediaPlayer1.play()
        self.mediaPlayer2.play()


def main():
    parser = argparse.ArgumentParser(description='Run the Video Survey Application.')
    parser.add_argument('--dummy_run', action='store_true', help='Run the application in dummy mode')
    parser.add_argument('--gen_id', required=True, type=int, help='enter the generation id')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    if args.dummy_run:
        survey_app = VideoSurveyApp(dummy_run=args.dummy_run, gen_id=args.gen_id)
    else:
        survey_app = VideoSurveyApp(dummy_run=args.dummy_run, gen_id=args.gen_id)

    survey_app.show()
    # survey_app.play_videos()

    # survey_app.completed.connect(app.quit)  # Quit app when completed

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
