# Ubuntu Core – Text Classifier na Raspberry Pi

Ovaj dokument opisuje korake za izgradnju Ubuntu Core imagea koji uključuje
`textclassifier` snap aplikaciju i pokretanje na Raspberry Pi uređaju.

## Preduvjeti

- Ubuntu ili WSL2 okruženje
- Instaliran `snapcraft`: `sudo snap install snapcraft --classic`
- Instaliran `ubuntu-image`: `sudo snap install ubuntu-image --classic --edge`
- Raspberry Pi (arm64, npr. Raspberry Pi 4 ili 5)
- SD kartica s minimalno 4 GB prostora

## Korak 1 – Izgradnja snap paketa za arm64

Iz root-a projekta pokrenite:

```bash
snapcraft --build-for arm64
```

Rezultat je datoteka `textclassifier_1.0.0_arm64.snap`. Premjestite je u
`ubuntu-core/` direktorij:

```bash
mv textclassifier_1.0.0_arm64.snap ubuntu-core/
```

## Korak 2 – Generiranje Ubuntu Core imagea

Prijeđite u `ubuntu-core/` direktorij:

```bash
cd ubuntu-core
```

Budući da model koristi `grade: dangerous`, potpisivanje nije potrebno.
Generirajte image izravno:

```bash
ubuntu-image snap --allow-snapd-kernel-mismatch textclassifier.json \
    --snap textclassifier_1.0.0_arm64.snap
```

Rezultat je datoteka `.img` spremna za zapisivanje na SD karticu.

## Korak 3 – Zapisivanje imagea na SD karticu

Koristite jedan od alata:

**balenaEtcher** (grafički, preporučeno):
- Preuzmite s https://etcher.balena.io/
- Odaberite `.img` datoteku i SD karticu, kliknite Flash.

**Raspberry Pi Imager** (grafički):
- Odaberite "Use custom image" i odaberite generiranu `.img` datoteku.

**dd** (terminal, Linux):
```bash
sudo dd if=pc.img of=/dev/sdX bs=4M status=progress
```
Zamijenite `/dev/sdX` s ispravnim nazivom vaše SD kartice (`lsblk` za provjeru).

## Korak 4 – Pokretanje na uređaju

1. Umetnite SD karticu u Raspberry Pi.
2. Spojite tipkovnicu, monitor i mrežni kabel.
3. Uključite uređaj i pratite upute `console-conf` za inicijalnu konfiguraciju
   (unos Ubuntu One korisničkog računa i SSH ključa).
4. Spojite se SSH-om:
   ```bash
   ssh <ubuntu-one-korisnik>@<ip-adresa>
   ```
5. Pokrenite aplikaciju:
   ```bash
   textclassifier "Manchester United will face Arsenal on Saturday."
   ```

## Struktura direktorija

```
ubuntu-core/
├── textclassifier.json          # Ubuntu Core model assertion
├── textclassifier_1.0.0_arm64.snap  # snap paket (generiran, nije u gitu)
└── README.md
```

> **Napomena:** `.snap` datoteka nije commitana u repozitorij zbog veličine.
> Izgradite je lokalno prema Koraku 1.
