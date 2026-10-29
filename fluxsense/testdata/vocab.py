# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/testdata/vocab.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Built-in fictional vocabulary of the test-document generator:
                placeholder person names (Muster-/Beispiel-/Probe-style
                surnames), invented companies, towns and streets, generic
                article and booking texts. Written for this project; no
                external name lists and no real documents were used.
------------------------------------------------------------------------------
"""

from typing import Dict, Tuple

FIRST_NAMES: Tuple[str, ...] = (
    "Erika", "Max", "Hans", "Anna", "Lena", "Jonas", "Paul", "Clara", "Felix", "Mia", "Lukas", "Greta",
    "Emil", "Ida", "Oskar", "Frieda", "Theo", "Lotte", "Moritz", "Hanna")

# Surnames that read as placeholders, so a generated person is never mistaken for a real one.
SURNAMES: Tuple[str, ...] = (
    "Mustermann", "Musterfrau", "Beispiel", "Probe", "Testmann", "Exempel", "Platzhalter", "Vorlage",
    "Schablone", "Muster", "Probst-Beispiel", "Musterhaus", "Testfeld", "Beispielmann")

SALUTATIONS: Dict[str, Tuple[str, str]] = {"f": ("Frau", "Sehr geehrte Frau"), "m": ("Herrn", "Sehr geehrter Herr")}
FEMALE: Tuple[str, ...] = ("Erika", "Anna", "Lena", "Clara", "Mia", "Greta", "Ida", "Frieda", "Lotte", "Hanna")

TOWN_STEMS: Tuple[str, ...] = ("Muster", "Beispiel", "Probe", "Exempel", "Vorlagen", "Platzhalter", "Schablonen")
TOWN_SUFFIXES: Tuple[str, ...] = ("stadt", "dorf", "hausen", "heim", "feld", "berg", "burg", "tal")

STREETS: Tuple[str, ...] = (
    "Musterstraße", "Beispielweg", "Probegasse", "Am Testfeld", "Vorlagenring", "Exempelplatz", "Platzhalterallee",
    "Schablonenweg", "Musterdamm", "Beispielufer", "Probepfad", "Testgasse")

COMPANY_STEMS: Tuple[str, ...] = (
    "Musterwerk", "Beispiel Elektro", "Probe Logistik", "Exempel Software", "Vorlage Druck", "Schablone Bau",
    "Platzhalter Handel", "Muster Haustechnik", "Beispiel Bauteile", "Probe Energie", "Testfeld Mobilfunk",
    "Exempel Versand", "Muster Kfz-Service", "Beispiel Immobilien")
LEGAL_FORMS: Tuple[str, ...] = ("GmbH", "AG", "KG", "eG", "GmbH & Co. KG", "UG (haftungsbeschränkt)", "SE")

BANKS: Tuple[str, ...] = ("Beispielbank AG", "Musterbank eG", "Probe Kreditbank AG", "Exempel Direktbank AG")
CARD_ISSUERS: Tuple[str, ...] = ("Beispiel Card Services GmbH", "Muster Kartenbank AG")
SHOPS: Tuple[str, ...] = ("Beispiel Baumarkt", "Muster Getränkemarkt", "Probe Bürobedarf", "Exempel Gartencenter")
AUTHORITIES: Tuple[str, ...] = ("Stadtverwaltung", "Bürgeramt", "Ordnungsamt")
INSURERS: Tuple[str, ...] = ("Beispiel Versicherung AG", "Muster Lebensversicherung a.G.", "Probe Krankenkasse")

# (German description, English description, unit price range in cents)
ARTICLES: Tuple[Tuple[str, str, Tuple[int, int]], ...] = (
    ("Kabelbinder 200 mm, 100 Stück", "Cable ties 200 mm, pack of 100", (199, 899)),
    ("Schraube M4x12 Edelstahl", "Screw M4x12 stainless steel", (5, 40)),
    ("Widerstand 10 kOhm 0,25 W", "Resistor 10 kOhm 0.25 W", (2, 30)),
    ("Kondensator 100 nF 50 V", "Capacitor 100 nF 50 V", (3, 45)),
    ("Steckverbinder 2-polig", "Connector 2-pin", (15, 190)),
    ("Leuchtdiode grün 5 mm", "LED green 5 mm", (4, 60)),
    ("Netzteil 12 V 2 A", "Power supply 12 V 2 A", (890, 2990)),
    ("Lötzinn 0,5 mm 100 g", "Solder wire 0.5 mm 100 g", (690, 1990)),
    ("Schrumpfschlauch-Set", "Heat-shrink tubing set", (390, 1490)),
    ("Sicherung 2 A träge", "Fuse 2 A slow-blow", (20, 120)),
    ("Montagezeit je Stunde", "Labour per hour", (4500, 8900)),
    ("Versandpauschale", "Shipping flat rate", (490, 990)),
    ("Gehäuse ABS 120x80x40", "Enclosure ABS 120x80x40", (450, 1890)),
    ("Klemmleiste 12-fach", "Terminal strip 12-way", (120, 690)),
    ("Spannungsregler 5 V", "Voltage regulator 5 V", (35, 290)),
    ("Taster rund schwarz", "Push button round black", (40, 350)),
)

RECEIPT_ITEMS: Tuple[Tuple[str, Tuple[int, int]], ...] = (
    ("Blumenerde 40 l", (399, 999)), ("Gartenschere", (899, 2499)), ("Holzleim 250 g", (349, 799)),
    ("Schleifpapier-Set", (299, 899)), ("Mineralwasser Kiste", (499, 799)), ("Druckerpapier A4", (399, 699)),
    ("Kugelschreiber 10er", (199, 599)), ("Pinsel-Set", (499, 1299)), ("Arbeitshandschuhe", (299, 999)))

BOOKING_TEXTS: Tuple[Tuple[str, int], ...] = (
    ("Lastschrift Probe Energie GmbH Abschlag", -1), ("Gutschrift Gehalt Musterwerk GmbH", 1),
    ("Kartenzahlung Beispiel Baumarkt", -1), ("Dauerauftrag Miete Beispiel Immobilien KG", -1),
    ("Überweisung Exempel Versand GmbH", -1), ("Lastschrift Testfeld Mobilfunk AG", -1),
    ("Gutschrift Erstattung Probe Krankenkasse", 1), ("Kartenzahlung Muster Getränkemarkt", -1),
    ("Bargeldauszahlung Automat", -1), ("Überweisung Vorlage Druck GmbH", -1))

CARD_TEXTS: Tuple[str, ...] = (
    "Online-Shop Exempel Versand", "Tankstelle Musterdamm", "Hotel Beispielhof", "Restaurant Probeküche",
    "Bahnticket Online", "Buchhandlung Vorlage", "Streamingdienst Monatsabo", "Parkhaus Musterplatz")

LETTER_SUBJECTS: Tuple[Tuple[str, str], ...] = (
    ("Ihre Anfrage vom {ref_date}", "wir bestätigen den Eingang Ihrer Anfrage und melden uns in Kürze."),
    ("Änderung Ihrer Vertragsdaten", "wir haben die gewünschte Änderung Ihrer Kontaktdaten vorgenommen."),
    ("Wartung der Heizungsanlage", "die jährliche Wartung der Heizungsanlage ist für Ihr Objekt geplant."),
    ("Anpassung des Abschlags", "aufgrund Ihres Verbrauchs passen wir den monatlichen Abschlag an."),
)

FORM_TITLES: Tuple[Tuple[str, str], ...] = (
    ("Antrag auf Erstattung", "Erstattungsbetrag"), ("Änderungsmitteilung", "Neue Anschrift ab"),
    ("Kündigung des Vertrags", "Kündigung zum"))

TERMS_TITLE = "Allgemeine Geschäftsbedingungen"
TERMS_LINES: Tuple[str, ...] = (
    "1. Geltungsbereich: Diese Bedingungen gelten für alle Leistungen des Anbieters.",
    "2. Zahlung: Rechnungen sind ohne Abzug innerhalb der genannten Frist zahlbar.",
    "3. Gewährleistung: Es gelten die gesetzlichen Bestimmungen.",
    "4. Datenschutz: Hinweise zur Verarbeitung Ihrer Daten erhalten Sie auf Anfrage.",
    "5. Gerichtsstand ist der Sitz des Anbieters, soweit gesetzlich zulässig.")
