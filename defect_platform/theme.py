# -*- coding: utf-8 -*-
"""Light industrial-inspection UI theme. Separate from inspection logic."""
LIGHT_THEME = """
QMainWindow, QWidget { background:#F3F6FB; color:#22334B; font-family:"Microsoft YaHei UI"; font-size:12px; }
QLabel { background:transparent; border:0; color:#3E516B; }
QLabel#pageTitle { font-size:21px; font-weight:800; color:#1C3150; }
QLabel#pageSub { font-size:12px; color:#77869E; }
QGroupBox { background:#FFFFFF; border:1px solid #DDE6F0; border-radius:12px; margin-top:16px;
            padding:12px 14px 10px 14px; font-weight:700; color:#25415E; }
QGroupBox::title { subcontrol-origin:margin; subcontrol-position:top left; left:14px;
                   padding:0 4px; background:transparent; }
QPushButton { background:white; border:1px solid #D5E0EB; border-radius:8px;
              padding:9px 12px; font-weight:600; color:#314967; }
QPushButton:hover { background:#E9F5FD; border-color:#80B6D9; }
QPushButton:disabled { background:#E9EEF4; color:#9AA8BC; }
QPushButton#primary { background:#207DC2; border:1px solid #207DC2; color:white; }
QPushButton#primary:hover { background:#1669AA; }
QPushButton#live { background:#E5F7F2; border:1px solid #99DDD0; color:#157A68; }
QPushButton#live:hover { background:#CFF2E7; }
QSlider::groove:horizontal { height:5px; background:#DAE4F0; border-radius:2px; }
QSlider::handle:horizontal { background:#2889C7; width:16px; margin:-5px 0; border-radius:8px; }
QComboBox { background:white; border:1px solid #D4DFEC; border-radius:7px; padding:5px; }
QTableWidget, QListWidget { background:#FFFFFF; alternate-background-color:#F6F9FC;
                          border:1px solid #E3EAF2; gridline-color:#E5ECF4;
                          selection-background-color:#DDEEFF; selection-color:#143A5E; }
QHeaderView::section { background:#EDF3F9; color:#475D77; border:none; padding:8px; font-weight:700; }
QRadioButton, QCheckBox { padding:4px; background:transparent; }
QProgressBar { background:#E3EBF3; border:0; border-radius:5px; color:#1B527B; min-height:9px; text-align:center; }
QProgressBar::chunk { background:#278EBC; border-radius:5px; }
QSplitter::handle { background:#E1E8F0; }
QScrollArea { background:transparent; border:0; }
QScrollArea > QWidget > QWidget { background:transparent; }
QScrollBar:vertical { background:transparent; width:8px; margin:0; }
QScrollBar::handle:vertical { background:#C7D6E5; border-radius:4px; min-height:32px; }
QScrollBar::handle:vertical:hover { background:#AFC4D8; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:transparent; }
"""
