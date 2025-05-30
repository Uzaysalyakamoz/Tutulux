# -*- coding: utf-8 -*-
"""
Created on Wed May 28 22:12:47 2025

@author: Casper
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Circle
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QMenuBar, QAction,
    QLabel, QFileDialog, QSpinBox, QGroupBox, QTextEdit, QGridLayout,
    QComboBox, QInputDialog, QMessageBox, QDialog, QPushButton, QWidgetAction
)
from PyQt5.QtCore import Qt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from scipy.optimize import curve_fit
import pandas as pd
from astroquery.vizier import Vizier
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.visualization import ZScaleInterval, MinMaxInterval, LinearStretch, LogStretch
import requests
import time
import warnings
import uuid

warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='astropy.wcs.wcs')

gozlemevleri = {
    'TUG': (36.824, 30.335),
    'Kreiken Gözlemevi': (39.925, 32.783),
}

# Translation dictionary for UI elements
translations = {
    'tr': {
        'window_title': 'TUTULUX',
        'file_menu': 'Dosya',
        'load_fits': 'FITS Dosyası Aç',
        'load_ref': 'Referans Görüntü Aç',
        'save_header': 'Header Kaydet (.tmp)',
        'star_analysis_menu': 'Yıldız Analizi',
        'select_mode': 'Yıldız Seç (Mod)',
        'select_mode_active': 'Yıldız Seç (Aktif)',
        'select_mode_inactive': 'Yıldız Seç (Pasif)',
        'clear_selection': 'Seçimi Temizle',
        'auto_detect': 'Otomatik Yıldız Tespiti',
        'save_coords': 'Koordinatları Kaydet',
        'bv_analysis': 'B-V Analizi',
        'catalog_menu': 'Katalog',
        'wcs_check': 'WCS Kontrolü',
        'apass_match': 'APASS Eşleştirme',
        'graphs_menu': 'Grafikler',
        'magnitude_hist': 'Magnitüd Histogramı',
        'apass_plot': 'APASS Grafiği',
        'fit_error': 'Fit ve Hata',
        'about_menu': 'Hakkında',
        'about_action': 'Program Hakkında',
        'settings_menu': 'Ayarlar',
        'language': 'Dil',
        'observatory_group': 'Gözlemevi Seçimi',
        'select_observatory': 'Gözlemevi Seç:',
        'add_observatory': 'Gözlemevi Ekle',
        'coords_label': 'Koordinatlar: Enlem - Boylam',
        'aperture_group': 'Apertür Yarıçapları',
        'star_radius': 'Yıldız Yarıçapı (piksel):',
        'empty_radius': 'Boş Alan Yarıçapı (piksel):',
        'sky_radius': 'Gökyüzü Yarıçapı (piksel):',
        'scale_group': 'Görüntü Ölçekleme',
        'scale_type': 'Ölçek Türü:',
        'vmin': 'Minimum Parlaklık:',
        'vmax': 'Maksimum Parlaklık:',
        'header_group': 'Header Bilgileri',
        'zoom_group': 'Seçilen Yıldızın Yakın Görüntüsü',
        'gauss_group': 'Yıldız Profili (Gauss Fit)',
        'results_group': 'Fotometri Sonuçları',
        'exit_button': 'Çıkış',
        'about_text': (
            "Tutulux\n\n"
            "Sürüm: 1.2\n\n"
            "Bu program, astronomik FITS görüntüleri üzerinde fotometri analizi yapmak için tasarlanmıştır. "
            "Ana özellikler:\n"
            "- FITS dosyalarından yıldız tespiti ve fotometri analizi\n"
            "- Otomatik yıldız tespiti (DAOStarFinder)\n"
            "- B-V renk analizi\n"
            "- APASS kataloğu ile yıldız eşleştirme\n"
            "- WCS kontrolü ve güncelleme (Astrometry.net entegrasyonu)\n"
            "- Magnitüd histogramları ve karşılaştırma grafikleri\n"
            "- Çoklu dil desteği (Türkçe ve İngilizce)\n\n"
            "Geliştirici: Emre Bilgin\n"
            "İletişim: emre.bilgin64@gmail.com\n"
            "GitHub: https://github.com/Uzaysalyakamoz/Tutulux"
            "Bu yazılım, astronomi topluluğuna katkı sağlamak amacıyla açık kaynak olarak geliştirilmiştir."
        ),
        'success': 'Başarılı',
        'error': 'Hata',
        'file_loaded': 'FITS dosyası yüklendi!',
        'ref_loaded': 'Referans görüntü yüklendi ve yıldızlar tespit edildi!',
        'coords_saved': 'Koordinatlar tum_yildiz_koordinatlari.csv\'ye kaydedildi',
        'header_saved': 'Header {filename}\'ya kaydedildi',
        'wcs_updated': 'WCS güncellendi',
        'stars_detected': '{count} yıldız tespit edildi',
        'apass_matched': '{count} yıldız APASS ile eşleşti',
        'bv_completed': 'B-V analizi tamamlandı ve sonuçlar kaydedildi!',
        'no_file': 'FITS dosyası yüklü değil!',
        'no_wcs_stars': 'Geçerli WCS veya yıldızlar yok!',
        'no_stars': 'Yıldızlar tespit edilmedi!',
        'no_apass_data': 'APASS kataloğunda veri bulunamadı!',
        'no_apass_match': 'APASS kataloğunda eşleşme bulunamadı!',
        'no_bv_data': 'Geçerli B-V verisi hesaplanamadı!',
        'api_key_required': 'API anahtarı gerekli!',
        'astrometry_failed': 'Astrometry.net başarısız: {error}',
        'api_request_failed': 'API isteği başarısız: HTTP {code}',
        'api_connection_error': 'API bağlantı hatası: {error}',
        'save_coords_error': 'Koordinatlar kaydedilemedi: {error}',
        'save_header_error': 'Dosya kaydedilemedi: {error}',
        'catalog_match_error': 'Katalog eşleştirme hatası: {error}',
        'bv_analysis_error': 'B-V analizi hatası: {error}',
        'plot_error': 'Grafik oluşturma hatası: {error}',
        'hist_error': 'Histogram grafik hatası: {error}',
        'fit_error': 'Magnitüd karşılaştırma hatası: {error}',
        'no_stars_detected': 'Hiç yıldız tespit edilmedi!',
        'no_photometry': 'Fotometri, WCS veya yıldız eksik!',
        'no_matched_stars': 'Eşleşen yıldız bulunamadı!',
        'fwhm_input': 'Yıldız tespiti için FWHM:',
        'threshold_input': 'Tespit eşiği:',
        'observatory_name': 'Gözlemevi adı:',
        'latitude_input': '{name} için enlem',
        'longitude_input': '{name} için boylam',
        'add_observatory_success': '{name} gözlemevi eklendi',
        'add_observatory_error': 'Gözlemevi eklenemedi: {error}',
        'astrometry_wait': 'WCS çözümü bekleniyor... Deneme {attempt}/10',
        'astrometry_upload': 'WCS çözümü için Astrometry.net\'e yükleniyor...',
        'wcs_not_solved': 'WCS çözümü alınamadı!'
    },
    'en': {
        'window_title': 'TUTULUX',
        'file_menu': 'File',
        'load_fits': 'Open FITS File',
        'load_ref': 'Open Reference Image',
        'save_header': 'Save Header (.tmp)',
        'star_analysis_menu': 'Star Analysis',
        'select_mode': 'Select Star (Mode)',
        'select_mode_active': 'Select Star (Active)',
        'select_mode_inactive': 'Select Star (Inactive)',
        'clear_selection': 'Clear Selection',
        'auto_detect': 'Automatic Star Detection',
        'save_coords': 'Save Coordinates',
        'bv_analysis': 'B-V Analysis',
        'catalog_menu': 'Catalog',
        'wcs_check': 'WCS Check',
        'apass_match': 'APASS Matching',
        'graphs_menu': 'Graphs',
        'magnitude_hist': 'Magnitude Histogram',
        'apass_plot': 'APASS Plot',
        'fit_error': 'Fit and Error',
        'about_menu': 'About',
        'about_action': 'About the Program',
        'settings_menu': 'Settings',
        'language': 'Language',
        'observatory_group': 'Observatory Selection',
        'select_observatory': 'Select Observatory:',
        'add_observatory': 'Add Observatory',
        'coords_label': 'Coordinates: Latitude - Longitude',
        'aperture_group': 'Aperture Radii',
        'star_radius': 'Star Radius (pixels):',
        'empty_radius': 'Empty Radius (pixels):',
        'sky_radius': 'Sky Radius (pixels):',
        'scale_group': 'Image Scaling',
        'scale_type': 'Scale Type:',
        'vmin': 'Minimum Brightness:',
        'vmax': 'Maximum Brightness:',
        'header_group': 'Header Information',
        'zoom_group': 'Selected Star Zoom',
        'gauss_group': 'Star Profile (Gaussian Fit)',
        'results_group': 'Photometry Results',
        'exit_button': 'Exit',
        'about_text': (
            "Tutulux\n\n"
            "Version: 1.2\n\n"
            "This program is designed for photometric analysis of astronomical FITS images. "
            "Key features:\n"
            "- Star detection and photometry from FITS files\n"
            "- Automatic star detection (DAOStarFinder)\n"
            "- B-V color analysis\n"
            "- Star matching with the APASS catalog\n"
            "- WCS verification and update (Astrometry.net integration)\n"
            "- Magnitude histograms and comparison plots\n"
            "- Multilingual support (Turkish and English)\n\n"
            "Developer: Emre Bilgin\n"
            "Contact: emre.bilgin64@gmail.com\n"
            "GitHub: https://github.com/Uzaysalyakamoz/Tutulux\n\n"
            "This software is developed as an open-source tool to contribute to the astronomy community."
        ),
        'success': 'Success',
        'error': 'Error',
        'file_loaded': 'FITS file loaded!',
        'ref_loaded': 'Reference image loaded and stars detected!',
        'coords_saved': 'Coordinates saved to tum_yildiz_koordinatlari.csv',
        'header_saved': 'Header saved to {filename}',
        'wcs_updated': 'WCS updated',
        'stars_detected': '{count} stars detected',
        'apass_matched': '{count} stars matched with APASS',
        'bv_completed': 'B-V analysis completed and results saved!',
        'no_file': 'No FITS file loaded!',
        'no_wcs_stars': 'No valid WCS or stars!',
        'no_stars': 'No stars detected!',
        'no_apass_data': 'No data found in APASS catalog!',
        'no_apass_match': 'No matches found in APASS catalog!',
        'no_bv_data': 'No valid B-V data calculated!',
        'api_key_required': 'API key required!',
        'astrometry_failed': 'Astrometry.net failed: {error}',
        'api_request_failed': 'API request failed: HTTP {code}',
        'api_connection_error': 'API connection error: {error}',
        'save_coords_error': 'Failed to save coordinates: {error}',
        'save_header_error': 'Failed to save file: {error}',
        'catalog_match_error': 'Catalog matching error: {error}',
        'bv_analysis_error': 'B-V analysis error: {error}',
        'plot_error': 'Plot creation error: {error}',
        'hist_error': 'Histogram plot error: {error}',
        'fit_error': 'Magnitude comparison error: {error}',
        'no_stars_detected': 'No stars detected!',
        'no_photometry': 'Missing photometry, WCS, or stars!',
        'no_matched_stars': 'No matched stars found!',
        'fwhm_input': 'FWHM for star detection:',
        'threshold_input': 'Detection threshold:',
        'observatory_name': 'Observatory name:',
        'latitude_input': 'Latitude for {name}',
        'longitude_input': 'Longitude for {name}',
        'add_observatory_success': '{name} observatory added',
        'add_observatory_error': 'Failed to add observatory: {error}',
        'astrometry_wait': 'Waiting for WCS solution... Attempt {attempt}/10',
        'astrometry_upload': 'Uploading to Astrometry.net for WCS solution...',
        'wcs_not_solved': 'WCS solution could not be obtained!'
    }
}

class FotometriAraciGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_language = 'tr'
        self.setWindowTitle(translations[self.current_language]['window_title'])
        self.setGeometry(100, 100, 1800, 1000)

        self.setStyleSheet("""
            QMainWindow { background-color: #F5F5F5; color: #333333; }
            QGroupBox { background-color: #FFFFFF; color: #333333; border: 1px solid #CCCCCC; border-radius: 5px; margin-top: 10px; padding: 10px; }
            QMenuBar { background-color: #4CAF50; color: #FFFFFF; }
            QMenuBar::item { background-color: #4CAF50; color: #FFFFFF; padding: 5px 10px; }
            QMenuBar::item:selected { background-color: #45a049; }
            QTextEdit { background-color: #FFFFFF; color: #333333; border: 1px solid #CCCCCC; border-radius: 5px; padding: 5px; }
            QLabel { color: #333333; }
            QSpinBox { background-color: #FFFFFF; color: #333333; border-radius: 5px; padding: 3px; }
            QComboBox { background-color: #FFFFFF; color: #333333; border-radius: 5px; padding: 3px; }
            QPushButton { background-color: #4CAF50; color: #FFFFFF; border-radius: 5px; padding: 5px; }
            QPushButton:hover { background-color: #45a049; }
        """)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)

        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.main_layout.addWidget(self.left_panel, stretch=3)

        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.main_layout.addWidget(self.right_panel, stretch=1)

        self.menu_bar = QMenuBar()
        self.setMenuBar(self.menu_bar)

        self.dosya_menu = self.menu_bar.addMenu(translations[self.current_language]['file_menu'])
        self.load_action = QAction(translations[self.current_language]['load_fits'], self)
        self.load_action.triggered.connect(self.fits_ac)
        self.load_ref_action = QAction(translations[self.current_language]['load_ref'], self)
        self.load_ref_action.triggered.connect(self.referans_goruntu_ac)
        self.save_header_action = QAction(translations[self.current_language]['save_header'], self)
        self.save_header_action.triggered.connect(self.baslik_kaydet)
        self.dosya_menu.addAction(self.load_action)
        self.dosya_menu.addAction(self.load_ref_action)
        self.dosya_menu.addAction(self.save_header_action)

        self.analiz_menu = self.menu_bar.addMenu(translations[self.current_language]['star_analysis_menu'])
        self.select_mode_action = QAction(translations[self.current_language]['select_mode'], self)
        self.select_mode_action.setCheckable(True)
        self.select_mode_action.triggered.connect(self.yildiz_sec_modu)
        self.clear_selection_action = QAction(translations[self.current_language]['clear_selection'], self)
        self.clear_selection_action.triggered.connect(self.secimi_temizle)
        self.auto_detect_action = QAction(translations[self.current_language]['auto_detect'], self)
        self.auto_detect_action.triggered.connect(self.otomatik_yildiz_tespiti)
        self.save_sources_action = QAction(translations[self.current_language]['save_coords'], self)
        self.save_sources_action.triggered.connect(self.koordinatlari_kaydet)
        self.bv_action = QAction(translations[self.current_language]['bv_analysis'], self)
        self.bv_action.triggered.connect(self.bv_analizi)
        self.analiz_menu.addAction(self.select_mode_action)
        self.analiz_menu.addAction(self.clear_selection_action)
        self.analiz_menu.addAction(self.auto_detect_action)
        self.analiz_menu.addAction(self.save_sources_action)
        self.analiz_menu.addAction(self.bv_action)

        self.katalog_menu = self.menu_bar.addMenu(translations[self.current_language]['catalog_menu'])
        self.wcs_check_action = QAction(translations[self.current_language]['wcs_check'], self)
        self.wcs_check_action.triggered.connect(self.wcs_kontrol)
        self.apass_match_action = QAction(translations[self.current_language]['apass_match'], self)
        self.apass_match_action.triggered.connect(self.katalog_eslestir)
        self.katalog_menu.addAction(self.wcs_check_action)
        self.katalog_menu.addAction(self.apass_match_action)

        self.grafik_menu = self.menu_bar.addMenu(translations[self.current_language]['graphs_menu'])
        self.histogram_action = QAction(translations[self.current_language]['magnitude_hist'], self)
        self.histogram_action.triggered.connect(self.magnitud_histogrami)
        self.plot_action = QAction(translations[self.current_language]['apass_plot'], self)
        self.plot_action.triggered.connect(self.apass_grafigi)
        self.fit_action = QAction(translations[self.current_language]['fit_error'], self)
        self.fit_action.triggered.connect(self.fit_ve_hata)
        self.grafik_menu.addAction(self.histogram_action)
        self.grafik_menu.addAction(self.plot_action)
        self.grafik_menu.addAction(self.fit_action)

        self.settings_menu = self.menu_bar.addMenu(translations[self.current_language]['settings_menu'])
        self.language_combo = QComboBox()
        self.language_combo.addItems(['Türkçe', 'English'])
        self.language_combo.currentIndexChanged.connect(self.change_language)
        self.language_action = QWidgetAction(self)
        self.language_action.setDefaultWidget(self.language_combo)
        self.settings_menu.addAction(self.language_action)

        self.hakkinda_menu = self.menu_bar.addMenu(translations[self.current_language]['about_menu'])
        self.hakkinda_action = QAction(translations[self.current_language]['about_action'], self)
        self.hakkinda_action.triggered.connect(self.hakkinda_goster)
        self.hakkinda_menu.addAction(self.hakkinda_action)

        self.figure = plt.figure(figsize=(12, 10))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.left_layout.addWidget(self.canvas)

        self.observatory_group = QGroupBox(translations[self.current_language]['observatory_group'])
        self.observatory_layout = QVBoxLayout()
        self.observatory_combo = QComboBox()
        self.observatory_combo.addItems(gozlemevleri.keys())
        self.observatory_combo.currentIndexChanged.connect(self.gozlemevi_koordinat_guncelle)
        self.observatory_combo.setToolTip(translations[self.current_language]['select_observatory'])
        self.add_observatory_button = QPushButton(translations[self.current_language]['add_observatory'])
        self.add_observatory_button.clicked.connect(self.gozlemevi_ekle)
        self.observatory_layout.addWidget(QLabel(translations[self.current_language]['select_observatory']))
        self.observatory_layout.addWidget(self.observatory_combo)
        self.observatory_layout.addWidget(self.add_observatory_button)
        self.observatory_group.setLayout(self.observatory_layout)
        self.coord_label = QLabel(translations[self.current_language]['coords_label'])
        self.left_layout.addWidget(self.observatory_group)
        self.left_layout.addWidget(self.coord_label)

        self.scale_group = QGroupBox(translations[self.current_language]['scale_group'])
        self.scale_layout = QGridLayout()
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(['zscale', 'minmax', 'linear', 'log'])
        self.scale_combo.currentIndexChanged.connect(self.update_image_scale)
        self.vmin_spin = QSpinBox()
        self.vmin_spin.setRange(-10000, 10000)
        self.vmin_spin.setValue(0)
        self.vmin_spin.valueChanged.connect(self.update_image_scale)
        self.vmax_spin = QSpinBox()
        self.vmax_spin.setRange(-10000, 10000)
        self.vmax_spin.setValue(1000)
        self.vmax_spin.valueChanged.connect(self.update_image_scale)
        self.scale_layout.addWidget(QLabel(translations[self.current_language]['scale_type']), 0, 0)
        self.scale_layout.addWidget(self.scale_combo, 0, 1)
        self.scale_layout.addWidget(QLabel(translations[self.current_language]['vmin']), 1, 0)
        self.scale_layout.addWidget(self.vmin_spin, 1, 1)
        self.scale_layout.addWidget(QLabel(translations[self.current_language]['vmax']), 2, 0)
        self.scale_layout.addWidget(self.vmax_spin, 2, 1)
        self.scale_group.setLayout(self.scale_layout)
        self.left_layout.addWidget(self.scale_group)

        self.radius_group = QGroupBox(translations[self.current_language]['aperture_group'])
        self.radius_layout = QGridLayout()
        self.star_radius_spin = QSpinBox()
        self.star_radius_spin.setValue(20)
        self.star_radius_spin.setRange(1, 100)
        self.star_radius_spin.valueChanged.connect(self.cercleri_guncelle)
        self.empty_radius_spin = QSpinBox()
        self.empty_radius_spin.setValue(25)
        self.empty_radius_spin.setRange(1, 100)
        self.empty_radius_spin.valueChanged.connect(self.cercleri_guncelle)
        self.sky_radius_spin = QSpinBox()
        self.sky_radius_spin.setValue(30)
        self.sky_radius_spin.setRange(1, 100)
        self.sky_radius_spin.valueChanged.connect(self.cercleri_guncelle)
        self.radius_layout.addWidget(QLabel(translations[self.current_language]['star_radius']), 0, 0)
        self.radius_layout.addWidget(self.star_radius_spin, 0, 1)
        self.radius_layout.addWidget(QLabel(translations[self.current_language]['empty_radius']), 1, 0)
        self.radius_layout.addWidget(self.empty_radius_spin, 1, 1)
        self.radius_layout.addWidget(QLabel(translations[self.current_language]['sky_radius']), 2, 0)
        self.radius_layout.addWidget(self.sky_radius_spin, 2, 1)
        self.radius_group.setLayout(self.radius_layout)
        self.left_layout.addWidget(self.radius_group)

        self.header_group = QGroupBox(translations[self.current_language]['header_group'])
        self.header_layout = QVBoxLayout()
        self.header_text = QTextEdit()
        self.header_text.setReadOnly(True)
        self.header_layout.addWidget(self.header_text)
        self.header_group.setLayout(self.header_layout)
        self.left_layout.addWidget(self.header_group)

        self.exit_button = QPushButton(translations[self.current_language]['exit_button'])
        self.exit_button.clicked.connect(self.close)
        self.left_layout.addWidget(self.exit_button)

        self.zoom_group = QGroupBox(translations[self.current_language]['zoom_group'])
        self.zoom_layout = QVBoxLayout()
        self.zoom_figure = plt.figure(figsize=(4, 3))
        self.zoom_canvas = FigureCanvas(self.zoom_figure)
        self.zoom_layout.addWidget(self.zoom_canvas)
        self.zoom_group.setLayout(self.zoom_layout)
        self.right_layout.addWidget(self.zoom_group)

        self.gauss_group = QGroupBox(translations[self.current_language]['gauss_group'])
        self.gauss_layout = QVBoxLayout()
        self.gauss_figure = plt.figure(figsize=(4, 3))
        self.gauss_canvas = FigureCanvas(self.gauss_figure)
        self.gauss_layout.addWidget(self.gauss_canvas)
        self.gauss_group.setLayout(self.gauss_layout)
        self.right_layout.addWidget(self.gauss_group)

        self.results_group = QGroupBox(translations[self.current_language]['results_group'])
        self.results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_layout.addWidget(self.results_text)
        self.results_group.setLayout(self.results_layout)
        self.right_layout.addWidget(self.results_group)

        self.fits_dosyasi = None
        self.veri = None
        self.baslik = None
        self.wcs = None
        self.yildizlar = []
        self.fotometri_tablosu = None
        self.katalog = None
        self.secili_yildiz = None
        self.select_mode = False
        self.cercler = []
        self.ref_fits_dosyasi = None
        self.ref_veri = None
        self.ref_baslik = None
        self.ref_wcs = None

        self.canvas.mpl_connect('motion_notify_event', self.fare_hareket)
        self.canvas.mpl_connect('button_press_event', self.fare_tiklama)
        self.canvas.mpl_connect('scroll_event', self.fare_kaydirma)
        
        
        
    def closeEvent(self, event):
        plt.close('all')
        super().closeEvent(event)

    def change_language(self):
        lang_index = self.language_combo.currentIndex()
        self.current_language = 'tr' if lang_index == 0 else 'en'
        tr = translations[self.current_language]
        self.setWindowTitle(tr['window_title'])
        self.dosya_menu.setTitle(tr['file_menu'])
        self.load_action.setText(tr['load_fits'])
        self.load_ref_action.setText(tr['load_ref'])
        self.save_header_action.setText(tr['save_header'])
        self.analiz_menu.setTitle(tr['star_analysis_menu'])
        self.select_mode_action.setText(tr['select_mode_active'] if self.select_mode else tr['select_mode_inactive'])
        self.clear_selection_action.setText(tr['clear_selection'])
        self.auto_detect_action.setText(tr['auto_detect'])
        self.save_sources_action.setText(tr['save_coords'])
        self.bv_action.setText(tr['bv_analysis'])
        self.katalog_menu.setTitle(tr['catalog_menu'])
        self.wcs_check_action.setText(tr['wcs_check'])
        self.apass_match_action.setText(tr['apass_match'])
        self.grafik_menu.setTitle(tr['graphs_menu'])
        self.histogram_action.setText(tr['magnitude_hist'])
        self.plot_action.setText(tr['apass_plot'])
        self.fit_action.setText(tr['fit_error'])
        self.hakkinda_menu.setTitle(tr['about_menu'])
        self.hakkinda_action.setText(tr['about_action'])
        self.settings_menu.setTitle(tr['settings_menu'])
        self.observatory_group.setTitle(tr['observatory_group'])
        self.observatory_combo.setToolTip(tr['select_observatory'])
        self.add_observatory_button.setText(tr['add_observatory'])
        self.scale_group.setTitle(tr['scale_group'])
        self.radius_group.setTitle(tr['aperture_group'])
        self.header_group.setTitle(tr['header_group'])
        self.zoom_group.setTitle(tr['zoom_group'])
        self.gauss_group.setTitle(tr['gauss_group'])
        self.results_group.setTitle(tr['results_group'])
        self.exit_button.setText(tr['exit_button'])
        for i in range(self.scale_layout.count()):
            widget = self.scale_layout.itemAt(i).widget()
            if isinstance(widget, QLabel):
                if i == 0:
                    widget.setText(tr['scale_type'])
                elif i == 2:
                    widget.setText(tr['vmin'])
                elif i == 4:
                    widget.setText(tr['vmax'])
        for i in range(self.observatory_layout.count()):
            widget = self.observatory_layout.itemAt(i).widget()
            if isinstance(widget, QLabel):
                widget.setText(tr['select_observatory'])
        for i in range(self.radius_layout.count()):
            widget = self.radius_layout.itemAt(i).widget()
            if isinstance(widget, QLabel):
                if i == 0:
                    widget.setText(tr['star_radius'])
                elif i == 2:
                    widget.setText(tr['empty_radius'])
                elif i == 4:
                    widget.setText(tr['sky_radius'])
        self.gozlemevi_koordinat_guncelle()

    def hakkinda_goster(self):
        QMessageBox.information(self, translations[self.current_language]['about_action'],
                                translations[self.current_language]['about_text'])

    def update_image_scale(self):
     if self.veri is None:
        return
    # Sinyalleri engelle
     self.vmin_spin.blockSignals(True)
     self.vmax_spin.blockSignals(True)
     self.scale_combo.blockSignals(True)
    
     scale_type = self.scale_combo.currentText()
     vmin = self.vmin_spin.value()
     vmax = self.vmax_spin.value()
    
     if scale_type == 'zscale':
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(self.veri)
     elif scale_type == 'minmax':
        interval = MinMaxInterval()
        vmin, vmax = interval.get_limits(self.veri)
     elif scale_type == 'linear':
        stretch = LinearStretch()
     elif scale_type == 'log':
        stretch = LogStretch()
    
     self.ax.clear()
     if scale_type in ['zscale', 'minmax', 'linear']:
        self.im = self.ax.imshow(self.veri, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
     else:
        norm = LogStretch()(self.veri)
        norm = (norm - np.min(norm)) / (np.max(norm) - np.min(norm)) * (vmax - vmin) + vmin
        self.im = self.ax.imshow(norm, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    
     for yildiz in self.yildizlar:
        self.ax.plot(yildiz['xcentroid'], yildiz['ycentroid'], 'r+', markersize=10)
     if self.secili_yildiz:
        x, y = self.secili_yildiz
        r_yildiz = self.star_radius_spin.value()
        r_bos = self.empty_radius_spin.value()
        r_gok = self.sky_radius_spin.value()
        yildiz_cerc = Circle((x, y), r_yildiz, fill=False, color='red', label='Yıldız' if self.current_language == 'tr' else 'Star')
        bos_cerc = Circle((x, y), r_bos, fill=False, color='yellow', label='Boş' if self.current_language == 'tr' else 'Empty')
        gok_cerc = Circle((x, y), r_gok, fill=False, color='blue', label='Gökyüzü' if self.current_language == 'tr' else 'Sky')
        self.ax.add_patch(yildiz_cerc)
        self.ax.add_patch(bos_cerc)
        self.ax.add_patch(gok_cerc)
        self.ax.legend()
     self.figure.colorbar(self.im)
     self.canvas.draw()
    
    # Sinyalleri geri aç
     self.vmin_spin.blockSignals(False)
     self.vmax_spin.blockSignals(False)
     self.scale_combo.blockSignals(False)

    def fits_ac(self):
        dosya, _ = QFileDialog.getOpenFileName(self, translations[self.current_language]['load_fits'], '', 'FITS files (*.fits *.fit)')
        if dosya:
            self.fits_dosyasi = dosya
            with fits.open(dosya) as hdul:
                self.veri = hdul[0].data
                self.baslik = hdul[0].header
                if 'EPOCH' in self.baslik and isinstance(self.baslik['EPOCH'], str):
                    try:
                        self.baslik['EPOCH'] = float(self.baslik['EPOCH'].strip())
                    except ValueError:
                        del self.baslik['EPOCH']
            self.wcs = WCS(self.baslik, fix=True) if 'NAXIS1' in self.baslik else None
            self.yildizlar = []
            self.cercler = []
            header_info = "\n".join([f"{k}: {v}" for k, v in self.baslik.items()])
            self.header_text.setText(header_info)
            self.ax.clear()
            self.update_image_scale()
            self.sabit_xlim = self.ax.get_xlim()
            self.sabit_ylim = self.ax.get_ylim()
            self.canvas.draw()
            self.auto_detect_action.setEnabled(True)
            self.save_header_action.setEnabled(True)
            self.wcs_check_action.setEnabled(True)
            QMessageBox.information(self, translations[self.current_language]['success'],
                                    translations[self.current_language]['file_loaded'])

    def referans_goruntu_ac(self):
        dosya, _ = QFileDialog.getOpenFileName(self, translations[self.current_language]['load_ref'], '', 'FITS files (*.fits *.fit)')
        if dosya:
            self.ref_fits_dosyasi = dosya
            with fits.open(dosya) as hdul:
                self.ref_veri = hdul[0].data
                self.ref_baslik = hdul[0].header
            self.ref_wcs = WCS(self.ref_baslik, fix=True) if 'NAXIS1' in self.ref_baslik else None
            self.yildizlar = []
            self.otomatik_yildiz_tespiti(referans=True)
            QMessageBox.information(self, translations[self.current_language]['success'],
                                    translations[self.current_language]['ref_loaded'])

    def yildiz_sec_modu(self):
        self.select_mode = self.select_mode_action.isChecked()
        self.select_mode_action.setText(translations[self.current_language]['select_mode_active'] if self.select_mode else translations[self.current_language]['select_mode_inactive'])

    def secimi_temizle(self):
        self.yildizlar = []
        self.cercler = []
        self.secili_yildiz = None
        self.results_text.clear()
        self.gauss_figure.clear()
        self.zoom_figure.clear()
        self.ax.clear()
        if self.veri is not None:
           self.update_image_scale()
        self.canvas.draw()
        self.gauss_canvas.draw()
        self.zoom_canvas.draw()

    def fare_hareket(self, event):
        if event.inaxes and self.veri is not None:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < self.veri.shape[1] and 0 <= y < self.veri.shape[0]:
                deger = self.veri[y, x]
                self.coord_label.setText(f'X: {x} Y: {y} Değer: {deger:.2f}')

    def fare_tiklama(self, event):
        if event.inaxes and event.button == 1 and self.select_mode and self.veri is not None:
            x, y = int(event.xdata), int(event.ydata)
            self.secili_yildiz = (x, y)
            self.yildizlar.append({'xcentroid': x, 'ycentroid': y})
            zoom_boyut = 200
            self.ax.set_xlim(x - zoom_boyut, x + zoom_boyut)
            self.ax.set_ylim(y - zoom_boyut, y + zoom_boyut)
            self.ax.plot(x, y, 'r+', markersize=10)
            self.cercleri_guncelle()
            self.canvas.draw()
            self.zoom_guncelle(x, y)
            self.yildiz_analiz_et(x, y)

    def fare_kaydirma(self, event):
        if event.inaxes:
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            xdata = event.xdata
            ydata = event.ydata
            scale_factor = 0.9 if event.button == 'up' else 1.1
            self.ax.set_xlim([xdata - (xdata - cur_xlim[0]) * scale_factor,
                              xdata + (cur_xlim[1] - xdata) * scale_factor])
            self.ax.set_ylim([ydata - (ydata - cur_ylim[0]) * scale_factor,
                              ydata + (cur_ylim[1] - ydata) * scale_factor])
            self.canvas.draw()

    def cercleri_guncelle(self):
        if self.secili_yildiz is None:
            return
        x, y = self.secili_yildiz
        self.ax.clear()
        self.update_image_scale()
        for yildiz in self.yildizlar:
            self.ax.plot(yildiz['xcentroid'], yildiz['ycentroid'], 'r+', markersize=10)
        r_yildiz = self.star_radius_spin.value()
        r_bos = self.empty_radius_spin.value()
        r_gok = self.sky_radius_spin.value()
        yildiz_cerc = Circle((x, y), r_yildiz, fill=False, color='red', label='Yıldız' if self.current_language == 'tr' else 'Star')
        bos_cerc = Circle((x, y), r_bos, fill=False, color='yellow', label='Boş' if self.current_language == 'tr' else 'Empty')
        gok_cerc = Circle((x, y), r_gok, fill=False, color='blue', label='Gökyüzü' if self.current_language == 'tr' else 'Sky')
        self.ax.add_patch(yildiz_cerc)
        self.ax.add_patch(bos_cerc)
        self.ax.add_patch(gok_cerc)
        self.ax.legend()
        self.canvas.draw()

    def zoom_guncelle(self, x, y):
        if self.veri is None:
            return
        zoom_boyut = 100
        x_min = max(0, x - zoom_boyut)
        x_max = min(self.veri.shape[1], x + zoom_boyut)
        y_min = max(0, y - zoom_boyut)
        y_max = min(self.veri.shape[0], y + zoom_boyut)
        zoom_veri = self.veri[y_min:y_max, x_min:x_max]
        self.zoom_figure.clear()
        ax_zoom = self.zoom_figure.add_subplot(111)
        scale_type = self.scale_combo.currentText()
        vmin = self.vmin_spin.value()
        vmax = self.vmax_spin.value()
        if scale_type == 'zscale':
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(zoom_veri)
        elif scale_type == 'minmax':
            interval = MinMaxInterval()
            vmin, vmax = interval.get_limits(zoom_veri)
        ax_zoom.imshow(zoom_veri, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        r_yildiz = self.star_radius_spin.value()
        r_bos = self.empty_radius_spin.value()
        r_gok = self.sky_radius_spin.value()
        merkez_x = x - x_min
        merkez_y = y - y_min
        yildiz_cerc = Circle((merkez_x, merkez_y), r_yildiz, fill=False, color='red', label='Yıldız' if self.current_language == 'tr' else 'Star')
        bos_cerc = Circle((merkez_x, merkez_y), r_bos, fill=False, color='yellow', label='Boş' if self.current_language == 'tr' else 'Empty')
        gok_cerc = Circle((merkez_x, merkez_y), r_gok, fill=False, color='blue', label='Gökyüzü' if self.current_language == 'tr' else 'Sky')
        ax_zoom.add_patch(yildiz_cerc)
        ax_zoom.add_patch(bos_cerc)
        ax_zoom.add_patch(gok_cerc)
        ax_zoom.set_title(translations[self.current_language]['zoom_group'])
        self.zoom_canvas.draw()

    def yildiz_analiz_et(self, x, y, veri=None):
        veri = veri if veri is not None else self.veri
        if veri is None:
            return None
        r_yildiz = self.star_radius_spin.value()
        r_bos = self.empty_radius_spin.value()
        r_gok = self.sky_radius_spin.value()
        apertur = CircularAperture((x, y), r=r_yildiz)
        halka = CircularAnnulus((x, y), r_in=r_bos, r_out=r_gok)
        fot_tablo = aperture_photometry(veri, apertur)
        halka_maskeler = halka.to_mask(method='center')
        gok_veri = halka_maskeler.multiply(veri)
        gok_ortalama = np.nanmean(gok_veri[gok_veri > 0]) if np.any(gok_veri > 0) else 0
        yildiz_akisi = fot_tablo['aperture_sum'][0]
        gok_akisi = gok_ortalama * apertur.area
        if yildiz_akisi - gok_akisi > 0:
            magnitud = -2.5 * np.log10(yildiz_akisi - gok_akisi)
        else:
            magnitud = float('nan')
        gurultu = np.sqrt(yildiz_akisi + gok_akisi)
        sinyal_gurultu = (yildiz_akisi - gok_akisi) / gurultu if yildiz_akisi - gok_akisi > 0 else 0

        y_veri = veri[int(y - r_yildiz):int(y + r_yildiz), int(x - r_yildiz):int(x + r_yildiz)]
        x_veri = np.arange(len(y_veri))
        def gaussian(x, a, mu, sigma):
            return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        try:
            popt, _ = curve_fit(gaussian, x_veri, y_veri.mean(axis=0))
            self.gauss_figure.clear()
            ax_gauss = self.gauss_figure.add_subplot(111)
            ax_gauss.plot(x_veri, y_veri.mean(axis=0), 'b.', label='Veri' if self.current_language == 'tr' else 'Data')
            ax_gauss.plot(x_veri, gaussian(x_veri, *popt), 'r-', label='Gaussian Fit')
            ax_gauss.set_title(translations[self.current_language]['gauss_group'])
            ax_gauss.legend()
            self.gauss_canvas.draw()
            fwhm = 2.355 * abs(popt[2])
        except RuntimeError:
            fwhm = 0
            self.gauss_figure.clear()
            ax_gauss = self.gauss_figure.add_subplot(111)
            ax_gauss.text(0.5, 0.5, "Gaussian fit başarısız" if self.current_language == 'tr' else "Gaussian fit failed", ha='center', va='center')
            self.gauss_canvas.draw()

        magnitud_str = f"{magnitud:.2f}" if not np.isnan(magnitud) else "Hesaplanamadı" if self.current_language == 'tr' else "Not calculated"
        sonuclar = (f"{translations[self.current_language]['results_group']}:\n\n"
                    f"Koordinatlar: x={x}, y={y}\n"
                    f"Magnitüd: {magnitud_str}\n"
                    f"Sinyal/Gürültü Oranı: {sinyal_gurultu:.2f}\n"
                    f"FWHM: {fwhm:.2f} piksel\n"
                    f"Yıldız Akısı: {yildiz_akisi:.2f}\n"
                    f"Gökyüzü Akısı: {gok_akisi:.2f}\n"
                    f"Net Akı: {yildiz_akisi - gok_akisi:.2f}\n\n"
                    f"Apertür Yarıçapları:\n"
                    f"Yıldız: {r_yildiz} piksel\n"
                    f"Boş Alan: {r_bos} piksel\n"
                    f"Gökyüzü: {r_gok} piksel\n\n"
                    f"Gaussian Fit Parametreleri:\n"
                    f"Genlik: {popt[0]:.2f}\n"
                    f"Merkez: {popt[1]:.2f}\n"
                    f"Sigma: {popt[2]:.2f}" if fwhm else "Gaussian fit başarısız" if self.current_language == 'tr' else "Gaussian fit failed")
        self.results_text.setText(sonuclar)

        df = pd.DataFrame({
            'x': [x], 'y': [y], 'magnitud': [magnitud], 'SNR': [sinyal_gurultu], 'FWHM': [fwhm]
        })
        df.to_csv(f'yildiz_koordinatlari_{x}_{y}.csv', index=False)

        self.fotometri_tablosu = fot_tablo
        self.fotometri_tablosu['flux'] = yildiz_akisi - gok_akisi
        return magnitud

    def otomatik_yildiz_tespiti(self, referans=False):
        veri = self.ref_veri if referans else self.veri
        if veri is None:
            QMessageBox.critical(self, translations[self.current_language]['error'],
                                 translations[self.current_language]['no_file'])
            return
        fwhm, ok = QInputDialog.getDouble(self, translations[self.current_language]['fwhm_input'],
                                          translations[self.current_language]['fwhm_input'], 3.0, 0.1, 10.0)
        if not ok:
            return
        esik, ok = QInputDialog.getDouble(self, translations[self.current_language]['threshold_input'],
                                          translations[self.current_language]['threshold_input'], 5.0, 1.0, 100.0)
        if not ok:
            return
        mean, median, std = sigma_clipped_stats(veri, sigma=3.0)
        daofind = DAOStarFinder(fwhm=fwhm, threshold=esik * std)
        yildiz_tablosu = daofind(veri - median)
        if yildiz_tablosu is None:
            QMessageBox.critical(self, translations[self.current_language]['error'],
                                 translations[self.current_language]['no_stars_detected'])
            return
        self.yildizlar = [{'xcentroid': star['xcentroid'], 'ycentroid': star['ycentroid']} for star in yildiz_tablosu]
        self.ax.clear()
        self.update_image_scale()
        for yildiz in self.yildizlar:
            self.ax.plot(yildiz['xcentroid'], yildiz['ycentroid'], 'r+', markersize=10)
        self.canvas.draw()
        self.save_sources_action.setEnabled(True)
        self.histogram_action.setEnabled(True)
        QMessageBox.information(self, translations[self.current_language]['success'],
                                translations[self.current_language]['stars_detected'].format(count=len(self.yildizlar)))

    def koordinatlari_kaydet(self):
        if self.yildizlar:
            try:
                df = pd.DataFrame([{'x': y['xcentroid'], 'y': y['ycentroid'],
                                    'pixel_value': self.veri[int(y['ycentroid']), int(y['xcentroid'])]}
                                   for y in self.yildizlar])
                df.to_csv('tum_yildiz_koordinatlari.csv', index=False)
                QMessageBox.information(self, translations[self.current_language]['success'],
                                        translations[self.current_language]['coords_saved'])
            except Exception as e:
                QMessageBox.critical(self, translations[self.current_language]['error'],
                                     translations[self.current_language]['save_coords_error'].format(error=str(e)))

    def gozlemevi_koordinat_guncelle(self):
        gozlemevi = self.observatory_combo.currentText()
        if gozlemevi in gozlemevleri:
            enlem, boylam = gozlemevleri[gozlemevi]
            self.coord_label.setText(translations[self.current_language]['coords_label'] + f": Enlem = {enlem}, Boylam = {boylam}")

    def gozlemevi_ekle(self):
        ad, ok = QInputDialog.getText(self, translations[self.current_language]['observatory_name'],
                                      translations[self.current_language]['observatory_name'])
        if ok and ad:
            try:
                enlem, ok = QInputDialog.getDouble(self, translations[self.current_language]['latitude_input'].format(name=ad),
                                                   translations[self.current_language]['latitude_input'].format(name=ad), 0, -90, 90)
                if ok:
                    boylam, ok = QInputDialog.getDouble(self, translations[self.current_language]['longitude_input'].format(name=ad),
                                                        translations[self.current_language]['longitude_input'].format(name=ad), 0, -180, 180)
                    if ok:
                        gozlemevleri[ad] = (enlem, boylam)
                        self.observatory_combo.addItem(ad)
                        QMessageBox.information(self, translations[self.current_language]['success'],
                                                translations[self.current_language]['add_observatory_success'].format(name=ad))
            except Exception as e:
                QMessageBox.critical(self, translations[self.current_language]['error'],
                                     translations[self.current_language]['add_observatory_error'].format(error=str(e)))

    def baslik_kaydet(self):
        if self.baslik is not None and self.fits_dosyasi is not None:
            try:
                satirlar = self.header_text.toPlainText().split("\n")
                yeni_baslik = fits.Header()
                for satir in satirlar:
                    if ":" in satir:
                        key, value = satir.split(":", 1)
                        yeni_baslik[key.strip()] = value.strip()
                with fits.open(self.fits_dosyasi) as hdul:
                    hdul[0].header = yeni_baslik
                    yeni_dosya = self.fits_dosyasi.replace('.fits', '.tmp.fits')
                    hdul.writeto(yeni_dosya, overwrite=True)
                QMessageBox.information(self, translations[self.current_language]['success'],
                                        translations[self.current_language]['header_saved'].format(filename=yeni_dosya))
            except Exception as e:
                QMessageBox.critical(self, translations[self.current_language]['error'],
                                     translations[self.current_language]['save_header_error'].format(error=str(e)))

    def wcs_kontrol(self):
        if self.veri is None:
            QMessageBox.critical(self, translations[self.current_language]['error'],
                                 translations[self.current_language]['no_file'])
            return
        api_anahtari, ok = QInputDialog.getText(self, translations[self.current_language]['api_key_required'],
                                                translations[self.current_language]['api_key_required'])
        if not ok or not api_anahtari:
            QMessageBox.critical(self, translations[self.current_language]['error'],
                                 translations[self.current_language]['api_key_required'])
            return
        if not self.wcs or not self.wcs.is_celestial:
            url = "https://nova.astrometry.net/api/upload"
            try:
                files = {'file': open(self.fits_dosyasi, 'rb')}
                data = {'apikey': api_anahtari}
                QMessageBox.information(self, translations[self.current_language]['astrometry_upload'],
                                        translations[self.current_language]['astrometry_upload'])
                response = requests.post(url, files=files, data=data, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    if result['status'] == 'success':
                        sub_id = result['subid']
                        wcs_baslik = None
                        for attempt in range(10):
                            QMessageBox.information(self, translations[self.current_language]['astrometry_wait'].format(attempt=attempt+1),
                                                    translations[self.current_language]['astrometry_wait'].format(attempt=attempt+1))
                            wcs_response = requests.get(f"https://nova.astrometry.net/wcs_file/{sub_id}", timeout=10)
                            if wcs_response.status_code == 200:
                                wcs_data = wcs_response.json()
                                if wcs_data:
                                    wcs_baslik = wcs_data
                                    break
                            time.sleep(5)
                        if wcs_baslik:
                            self.baslik.update(wcs_baslik)
                            self.wcs = WCS(self.baslik, fix=True)
                            with fits.open(self.fits_dosyasi, mode='update') as hdul:
                                hdul[0].header.update(wcs_baslik)
                            QMessageBox.information(self, translations[self.current_language]['success'],
                                                    translations[self.current_language]['wcs_updated'])
                        else:
                            QMessageBox.critical(self, translations[self.current_language]['error'],
                                                 translations[self.current_language]['wcs_not_solved'])
                    else:
                        QMessageBox.critical(self, translations[self.current_language]['error'],
                                             translations[self.current_language]['astrometry_failed'].format(error=result.get('error', 'Unknown error')))
                else:
                    QMessageBox.critical(self, translations[self.current_language]['error'],
                                         translations[self.current_language]['api_request_failed'].format(code=response.status_code))
            except requests.exceptions.RequestException as e:
                QMessageBox.critical(self, translations[self.current_language]['error'],
                                     translations[self.current_language]['api_connection_error'].format(error=str(e)))
            except Exception as e:
                QMessageBox.critical(self, translations[self.current_language]['error'],
                                     translations[self.current_language]['api_connection_error'].format(error=str(e)))

    def katalog_eslestir(self):
        if not self.wcs or not self.yildizlar:
            QMessageBox.critical(self, translations[self.current_language]['error'],
                                 translations[self.current_language]['no_wcs_stars'])
            return
        try:
            x_koordinatlari = [y['xcentroid'] for y in self.yildizlar]
            y_koordinatlari = [y['ycentroid'] for y in self.yildizlar]
            sky_coords = self.wcs.pixel_to_world(x_koordinatlari, y_koordinatlari)
            v = Vizier(columns=['RAJ2000', 'DEJ2000', 'Vmag', 'e_Vmag', 'B-V', 'e_B-V'])
            result = v.query_region(sky_coords, radius=30*u.arcsec, catalog='II/336/apass9')
            if result:
                self.katalog = result[0]
                self.ax.clear()
                self.update_image_scale()
                for yildiz in self.yildizlar:
                    self.ax.plot(yildiz['xcentroid'], yildiz['ycentroid'], 'r+', markersize=10)
                apass_coords = SkyCoord(ra=self.katalog['RAJ2000'], dec=self.katalog['DEJ2000'], unit='deg')
                apass_pix = self.wcs.world_to_pixel(apass_coords)
                for x, y in zip(apass_pix[0], apass_pix[1]):
                    self.ax.plot(x, y, 'g+', markersize=10)
                self.canvas.draw()
                self.plot_action.setEnabled(True)
                self.fit_action.setEnabled(True)
                QMessageBox.information(self, translations[self.current_language]['success'],
                                        translations[self.current_language]['apass_matched'].format(count=len(self.katalog)))
            else:
                QMessageBox.critical(self, translations[self.current_language]['error'],
                                     translations[self.current_language]['no_apass_match'])
        except Exception as e:
            QMessageBox.critical(self, translations[self.current_language]['error'],
                                 translations[self.current_language]['catalog_match_error'].format(error=str(e)))

    def bv_analizi(self):
        if not self.yildizlar:
            QMessageBox.critical(self, translations[self.current_language]['error'],
                                 translations[self.current_language]['no_stars'])
            return
        b_dosya, _ = QFileDialog.getOpenFileName(self, translations[self.current_language]['bv_analysis'], '', 'FITS files (*.fits *.fit)')
        if not b_dosya:
            return
        v_dosya, _ = QFileDialog.getOpenFileName(self, translations[self.current_language]['bv_analysis'], '', 'FITS files (*.fits *.fit)')
        if not v_dosya:
            return
        try:
            with fits.open(b_dosya) as hdul_b:
                b_veri = hdul_b[0].data
            with fits.open(v_dosya) as hdul_v:
                v_veri = hdul_v[0].data
            b_magnitudler = []
            v_magnitudler = []
            bv_degerleri = []
            for yildiz in self.yildizlar:
                x, y = yildiz['xcentroid'], yildiz['ycentroid']
                b_mag = self.yildiz_analiz_et(x, y, veri=b_veri)
                v_mag = self.yildiz_analiz_et(x, y, veri=v_veri)
                if b_mag is not None and v_mag is not None and not np.isnan(b_mag) and not np.isnan(v_mag):
                    b_magnitudler.append(b_mag)
                    v_magnitudler.append(v_mag)
                    bv_degerleri.append(b_mag - v_mag)
            if bv_degerleri:
                df = pd.DataFrame({
                    'x': [y['xcentroid'] for y in self.yildizlar[:len(bv_degerleri)]],
                    'y': [y['ycentroid'] for y in self.yildizlar[:len(bv_degerleri)]],
                    'B_magnitud': b_magnitudler,
                    'V_magnitud': v_magnitudler,
                    'B-V': bv_degerleri
                })
                df.to_csv('bv_analizi_sonuclari.csv', index=False)
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                ax.scatter(bv_degerleri, v_magnitudler, color='blue', alpha=0.6)
                ax.set_xlabel('B-V')
                ax.set_ylabel('V Magnitüd' if self.current_language == 'tr' else 'V Magnitude')
                ax.set_title('B-V vs. V Magnitüd' if self.current_language == 'tr' else 'B-V vs. V Magnitude')
                ax.grid(True)
                dialog = QDialog(self)
                dialog.setWindowTitle(translations[self.current_language]['bv_analysis'])
                dialog.setGeometry(100, 100, 800, 600)
                layout = QVBoxLayout(dialog)
                canvas = FigureCanvas(fig)
                layout.addWidget(canvas)
                dialog.exec_()
                QMessageBox.information(self, translations[self.current_language]['success'],
                                        translations[self.current_language]['bv_completed'])
            else:
                QMessageBox.critical(self, translations[self.current_language]['error'],
                                     translations[self.current_language]['no_bv_data'])
        except Exception as e:
            QMessageBox.critical(self, translations[self.current_language]['error'],
                                 translations[self.current_language]['bv_analysis_error'].format(error=str(e)))

    def apass_grafigi(self):
        if not hasattr(self, 'apass_dialog') or not self.apass_dialog.isVisible():
         print("apass_grafigi called")
        if not self.wcs or not self.yildizlar:
            QMessageBox.critical(self, translations[self.current_language]['error'],
                                 translations[self.current_language]['no_wcs_stars'])
            return
        try:
            merkez = SkyCoord(ra=self.wcs.wcs.crval[0], dec=self.wcs.wcs.crval[1], unit='deg')
            v = Vizier(columns=['RAJ2000', 'DEJ2000'])
            result = v.query_region(merkez, radius=1*u.deg, catalog='II/336/apass9')
            if result:
                apass_yildizlar = result[0].to_pandas()
                tespit_ras = []
                tespit_decs = []
                for y in self.yildizlar:
                    sky_coord = self.wcs.pixel_to_world(y['xcentroid'], y['ycentroid'])
                    if isinstance(sky_coord, SkyCoord):
                        tespit_ras.append(sky_coord.ra.deg)
                        tespit_decs.append(sky_coord.dec.deg)
                apass_ras = apass_yildizlar['RAJ2000'].tolist()
                apass_decs = apass_yildizlar['DEJ2000'].tolist()
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                ax.scatter(tespit_ras, tespit_decs, color='blue', label='Tespit Edilen Yıldızlar' if self.current_language == 'tr' else 'Detected Stars', alpha=0.6)
                ax.scatter(apass_ras, apass_decs, color='red', label='APASS Yıldızları' if self.current_language == 'tr' else 'APASS Stars', alpha=0.6)
                ax.set_xlabel('RA (derece)' if self.current_language == 'tr' else 'RA (degrees)')
                ax.set_ylabel('DEC (derece)' if self.current_language == 'tr' else 'DEC (degrees)')
                ax.set_title('Tespit Edilen vs. APASS Yıldız Pozisyonları' if self.current_language == 'tr' else 'Detected vs. APASS Star Positions')
                ax.legend()
                ax.grid(True)
                dialog = QDialog(self)
                self.apass_dialog = dialog
                dialog.setWindowTitle(translations[self.current_language]['apass_plot'])
                dialog.setGeometry(100, 100, 800, 600)
                layout = QVBoxLayout(dialog)
                canvas = FigureCanvas(fig)
                layout.addWidget(canvas)
                dialog.exec_()
            else:
                QMessageBox.critical(self, translations[self.current_language]['error'],
                                     translations[self.current_language]['no_apass_data'])
        except Exception as e:
            QMessageBox.critical(self, translations[self.current_language]['error'],
                                 translations[self.current_language]['plot_error'].format(error=str(e)))

    def fit_ve_hata(self):
        if not hasattr(self, 'fit_dialog') or not self.fit_dialog.isVisible():
         print("fit_ve_hata called")
        if not self.wcs or not self.yildizlar or not self.fotometri_tablosu:
            QMessageBox.critical(self, translations[self.current_language]['error'],
                                 translations[self.current_language]['no_photometry'])
            return
        try:
            merkez = SkyCoord(ra=self.wcs.wcs.crval[0], dec=self.wcs.wcs.crval[1], unit='deg')
            v = Vizier(columns=['RAJ2000', 'DEJ2000', 'Vmag', 'e_Vmag'])
            result = v.query_region(merkez, radius=1*u.deg, catalog='II/336/apass9')
            if result:
                apass_yildizlar = result[0].to_pandas()
                tespit_magnitudler = []
                apass_magnitudler = []
                for yildiz in self.yildizlar:
                    sky_coord = self.wcs.pixel_to_world(yildiz['xcentroid'], yildiz['ycentroid'])
                    if isinstance(sky_coord, SkyCoord):
                        ra, dec = sky_coord.ra.deg, sky_coord.dec.deg
                        for _, apass_row in apass_yildizlar.iterrows():
                            if np.isclose(ra, apass_row['RAJ2000'], atol=1e-3) and np.isclose(dec, apass_row['DEJ2000'], atol=1e-3):
                                flux = self.fotometri_tablosu['flux'][0]
                                mag = -2.5 * np.log10(flux) if flux > 0 else np.nan
                                if not np.isnan(mag):
                                    tespit_magnitudler.append(mag)
                                    apass_magnitudler.append(apass_row['Vmag'])
                                break
                if not tespit_magnitudler:
                    QMessageBox.critical(self, translations[self.current_language]['error'],
                                         translations[self.current_language]['no_matched_stars'])
                    return
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                ax.scatter(tespit_magnitudler, apass_magnitudler, color='blue',
                           label='Tespit vs. APASS Magnitüd' if self.current_language == 'tr' else 'Detected vs. APASS Magnitude', alpha=0.6)
                def linear_fit(x, m, b):
                    return m * x + b
                try:
                    popt, _ = curve_fit(linear_fit, tespit_magnitudler, apass_magnitudler)
                    fit_x = np.linspace(min(tespit_magnitudler), max(tespit_magnitudler), 100)
                    ax.plot(fit_x, linear_fit(fit_x, *popt), 'r-', label='Fit Çizgisi' if self.current_language == 'tr' else 'Fit Line')
                    residuals = np.array(apass_magnitudler) - np.array([linear_fit(x, *popt) for x in tespit_magnitudler])
                    hata = np.sqrt(np.mean(residuals ** 2))
                    ax.text(0.05, 0.95, f'Hata: {hata:.2f}' if self.current_language == 'tr' else f'Error: {hata:.2f}', transform=ax.transAxes, va='top')
                except RuntimeError:
                    ax.text(0.5, 0.5, "Fit başarısız" if self.current_language == 'tr' else "Fit failed", ha='center', va='center')
                ax.set_xlabel('Tespit Edilen Magnitüd' if self.current_language == 'tr' else 'Detected Magnitude')
                ax.set_ylabel('APASS Magnitüd' if self.current_language == 'tr' else 'APASS Magnitude')
                ax.set_title('Tespit vs. APASS Magnitüdler' if self.current_language == 'tr' else 'Detected vs. APASS Magnitudes')
                ax.legend()
                ax.grid(True)
                dialog = QDialog(self)
                self.fit_dialog = dialog
                dialog.setWindowTitle(translations[self.current_language]['fit_error'])
                dialog.setGeometry(100, 100, 800, 600)
                layout = QVBoxLayout(dialog)
                canvas = FigureCanvas(fig)
                layout.addWidget(canvas)
                dialog.exec_()
            else:
                QMessageBox.critical(self, translations[self.current_language]['error'],
                                     translations[self.current_language]['no_apass_data'])
        except Exception as e:
            QMessageBox.critical(self, translations[self.current_language]['error'],
                                 translations[self.current_language]['fit_error'].format(error=str(e)))

    def magnitud_histogrami(self):
        if not hasattr(self, 'hist_dialog') or not self.hist_dialog.isVisible():
         print("magnitud_histogrami called")
        if self.yildizlar and self.fotometri_tablosu:
            try:
                fluxlar = [self.fotometri_tablosu['flux']][0]
                magnituds = [-2.5 * np.log10(flux) for flux in fluxlar if flux > 0]
                if magnituds:
                    fig = plt.figure(figsize=(6, 4))
                    ax = fig.add_subplot(111)
                    ax.hist(magnituds, bins=20, color='blue', alpha=0.7)
                    ax.set_xlabel('Magnitüd' if self.current_language == 'tr' else 'Magnitude')
                    ax.set_ylabel('Yıldız Sayısı' if self.current_language == 'tr' else 'Number of Stars')
                    ax.set_title('Tespit Edilen Yıldız Magnitüd Histogramı' if self.current_language == 'tr' else 'Detected Star Magnitude Histogram')
                    dialog = QDialog(self)
                    self.hist_dialog = dialog
                    dialog.setWindowTitle(translations[self.current_language]['magnitude_hist'])
                    dialog.setGeometry(100, 100, 600, 400)
                    layout = QVBoxLayout(dialog)
                    canvas = FigureCanvas(fig)
                    layout.addWidget(canvas)
                    dialog.exec_()
            except Exception as e:
                QMessageBox.critical(self, translations[self.current_language]['error'],
                                     translations[self.current_language]['hist_error'].format(error=str(e)))

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = FotometriAraciGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
