# FITS Fotometri Analiz Programı

Bu program, astronomik FITS görüntüleri üzerinde fotometri analizi yapmak için geliştirilmiştir.

## Özellikler

- FITS dosyalarını açma ve görüntüleme
- Yıldız tespiti (manuel ve DAOStarFinder ile otomatik)
- APASS kataloğu ile eşleştirme
- B-V renk analizi
- WCS çözümü (Astrometry.net API desteği)
- Magnitüd histogramı ve grafikler
- Gauss profil analizi
- Çoklu dil desteği (TR/EN)

## Gereksinimler

- Python 3.8+
- Gerekli kütüphaneler için `requirements.txt` dosyasını kullanın:

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python ap-pho.py
```

## Notlar

- Astrometry.net API anahtarına ihtiyacınız olabilir (WCS çözümü için).
- `PyQt5` GUI kullanır, çalıştırmadan önce GUI desteği olan bir ortamda olduğunuzdan emin olun.

## Geliştirici

Emre Bilgin  
[emre.bilgin64@gmail.com](mailto:emre.bilgin64@gmail.com)  
[GitHub](https://github.com/Uzaysalyakamoz)
