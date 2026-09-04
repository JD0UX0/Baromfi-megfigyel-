# Baromfi Megfigyelő És Analitikai Rendszer (Desktop App)

A rendszer egy asztali alkalmazás (PyQt/Tkinter alapú GUI), amely videófolyamok és kameraképek alapján végzi a baromfik viselkedésének, egyedi aktivitásának, helyzetének és anomáliáinak valós idejű megfigyelését, logolását, valamint a megfigyelési zónák testreszabását.

---

## 1. Mappa- és Fájlstruktúra

* **`main.py`**: Az alkalmazás fő belépési pontja és a fő monitorozó felület (GUI) vezérlője (`Állataktivitás Monitor`).
* **`adjust_window.py`**: A megfigyelési zónák beállításáért felelős külön megnyíló ablak (`Megfigyelési zónák beállítása`).
* **`detector.py`**: Objektumdetekciós modul az egyedek, etetők és itatók azonosítására a képkockákon.
* **`tracker.py`**: A detektált egyedek mozgáskövetését (tracking) végző modul.
* **`anomaly.py`**: A szokatlan viselkedési minták, inaktivitás és anomáliák felismeréséért felelős modul.
* **`logger.py`**: Az események, aktivitások és mért adatok kiexportálását végzi (`.csv`) fájlokba.
* **`image_processor.py`**: Képfeldolgozási és transzformációs alapfunkciók (pl. képforgatás, eltolás).
* **`grid_overlay.py` / `rolling_grid.py`**: Vizualizációs elemek és rácsok megjelenítése.
* **`requirements.txt`**: A futtatáshoz szükséges függőségek listája.
* **`data/`**: Bemeneti videók és kimeneti analitikai adatok tárolóhelye.
* **`models/`**: Detekcióhoz és követséghez használt betanított modellek.
* **`parameters/`**: Rendszerkonfigurációs fájlok és mentett JSON zónabeállítások.

---

## 2. Főbb Funkciók és Modulok

### A) Fő Monitorozó Felület (`main.py`)
* **Videó betöltése & Vezérlés**: Tetszőleges bemeneti videófájl betöltése, FPS és aktív zónák számának kijelzése.
* **Élőkép és detektálás**: Valós idejű bounding box kijelölések az állatokra, etetőkre (sárga keret) és itatókra (kék keret).
* **Aktuális mérőszámok**: 
  * Összes detektált állat száma
  * Etetőnél / Itatónál tartózkodó egyedek száma
  * Átlagos mozgásintenzitás (px)
  * Alvó / inaktív egyedek száma
  * Detektált anomáliák száma
* **Egyedi állatfigyelő**: Kiválasztott egyed konkrét adatainak követése (mozgásintenzitás, etető/itató zónában tartózkodás, alvási állapot).
* **Idősoros grafikonom**: Dinamikus diagramok az állatszám és a zónahasználat időbeli alakulásáról.

### B) Zónaszerkesztő és Képforgatás (`adjust_window.py`)
* **Kép elforgatása**: A videókép tetszőleges szögben (0–360°) elforgatható a pontosabb perspektíva és zónakijelölés érdekében.
* **Zónák rajzolása és igazítása**: Etető (téglalap) és Itató (téglalap) zónák egérrel történő kijelölése, fine-tuning állítási lehetőségekkel (pozíció finomhangolása, szélesség/magasság állítása).
* **Konfigurációk mentése**: A felhasználó által megadott egyedi zóna- és forgatási beállítások JSON fájlokba mentődnek az adott kameranézethez/videóhoz.

### C) Adatmentés és Export
* **Excel logolás**: Az észlelt aktivitási adatok, zónastatisztikák és anomália-események Excelben is megnyitható CSV fájlokba exportálódnak a későbbi szakmai elemzésekhez.
