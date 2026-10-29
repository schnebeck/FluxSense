# FluxSense Field Contract

Version 1.3 (2026-10-29). This contract defines what each core field of a
logical document means. The engine (prompts, catalog, post-checks) and
any evaluation of its results follow it. Changing a rule here is a
deliberate decision and bumps the version.

## 0. General rules

1. **Printed or empty.** A field is filled only if its value is printed on the
   logical document's own pages. A field that is not printed stays empty;
   an empty field is never an error by itself.
2. **Copy, normalise, never compute.** Values are copied as printed and only
   their format is normalised: dates `YYYY-MM-DD`, amounts as decimals,
   IBANs compact upper case, identifiers trimmed of their label words
   (`Auszug 3` -> `3`, `Rech-Nr.: 880412-29` -> `880412-29`). Nothing is
   calculated, completed or taken from world knowledge.
3. **Stamps and handwriting are annotations.** Dates, numbers or names inside
   stamps (paid, received, `ÜBERWIESEN`, `Eingang`) or handwritten notes go
   to the visual audit, never into metadata fields. Exception: §2.4 forms.
4. **One label wins.** When the field has a label list below, a value next to
   a listed label beats any unlabelled value, and earlier labels in the list
   beat later ones.
5. **Provenance.** Every filled value records how it was determined:
   `label` (next to a listed label), `geometry` (layout region, e.g. the
   DIN 5008 address window), `check` (confirmed by a deterministic check),
   `model` (model reading only). Only `label`/`geometry`/`check` values that
   are also grounded in the text/OCR layer are auto-accepted; `model`-only
   values are flagged for review.

## 1. Logical documents (segmentation)

- A logical document is one item its sender issued as a unit: a letter with
  all its pages, a statement with its information/terms pages, an invoice
  with its continuation pages.
- Blank pages and boilerplate pages (terms, `Wichtige Informationen`,
  `Hinweise`, `Stand: MM/YY`) attach to the preceding document.
- Enclosures that are separate items (a copy of another letter, an app log,
  a protocol, a screenshot) are separate logical documents.
- Identical copies of one item in one file are separate documents.

## 2. Parties

`sender` = the party that issued the document. `recipient` = its addressee.
Names are given as printed: organisations by the name in the letterhead or
address block (departments optional); persons by first and last name
without salutation or title (`Herrn`, `Frau`, `Dr.` are dropped).
Salutations (`Sehr geehrte ...`), greetings, form headings
(`Privat Versicherte/ Selbstzahler`, `Patient/ Patientin`), table cells and
log entries are never parties.

| Case | sender | recipient |
|---|---|---|
| 2.1 Letter (DIN 5008 or similar) | letterhead, return-address line or signing organisation | the DIN 5008 address window; else the printed addressee |
| 2.2 Invoice, credit note, receipt | seller (BT-27) | buyer (BT-44) |
| 2.3 Bank / credit-card statement | bank or card issuer | account holder (address window) |
| 2.4 Form filled in by a person (Antrag, Formular, Kündigung, Vollmacht, Erklärung) | the person who fills in and signs (whoever it is; typed or handwritten entries count) | the organisation that issued and receives the form |
| 2.5 Pre-addressed cover or return sheet, not filled in | the issuing organisation | the printed address |
| 2.6 Enclosure without letterhead (log, protocol, screenshot, web page) | empty unless a sender is printed as such | empty unless printed |
| 2.7 Private notices, cards, notes (death notices, greeting cards, memos) | empty | empty |
| 2.8 Bank confirmation of a single transaction (Überweisungsbeleg, Auftragsbestätigung, Ausführungsbestätigung, Auslandsüberweisung, Einzahlungsbeleg, Lastschriftbeleg) | the bank that issued it | the account holder / ordering customer; the payee of the transfer is a counterparty, not the recipient |

`sender` and `recipient` are never the same party.

## 3. doc_date

| Type | doc_date | Labels (in priority order) |
|---|---|---|
| Invoice, credit note, dunning, utility bill | invoice date | `Rechnungsdatum`, `Re.-Dat`, `Liefer=Re.Dat`, `Invoice date`, `Datum` in the header block |
| Receipt | transaction/receipt date | `Datum`, receipt print date |
| Letter, official / legal correspondence | date of the letter | date line, info column `Datum`, `<Ort>, den <Datum>` |
| Bank / credit-card statement | statement or settlement date | `Auszugsdatum`, `Abrechnungsdatum`, `erstellt am`, `Rechnungsdatum` (credit-card statements only); else the printed period end |
| Form filled in by a person | date next to the signature (handwritten allowed) | `Datum`, `Ort, Datum` |
| Contract, policy, certificate, report | issue date | `ausgestellt am`, `Datum`, `Stand` of the document itself |
| Enclosure without letterhead | only if labelled as its own date | `Protokoll vom`, `Erstellt am` |

Never a doc_date: order date (`Auftragsdatum`), service/delivery date
(`Leistungsdatum`, `Lieferdatum`, `Datum fertig`), dates inside stamps,
transaction dates, form-version dates.

## 4. doc_number

| Type | doc_number | Labels (in priority order) |
|---|---|---|
| Invoice, credit note, utility bill | invoice / credit-note number (BT-1) | `Rechnungsnummer`, `Rech-Nr.`, `Rechnungs-Nr.`, `Gutschrift-Nr.`, `Invoice No` |
| Letter, official / legal correspondence | the sender's own reference | `Unser Zeichen`, `Aktenzeichen`, `Geschäftszeichen`, `Vorgangsnummer`, `Schadennummer`, `Vertragsnummer`, `Versicherungsschein-Nr.`, `Versicherungsnummer`, `Kundennummer` |
| Bank / credit-card statement | printed statement number | `Auszug`, `Kontoauszug Nr.`, `Abrechnung Nr.` |
| Form | the form's own number, else the number of the contract/order it refers to | `Antragsnummer`, `Versicherungsschein-Nr.`, `Vertragsnummer`, `Mitglieds-Nr.`, `Mitgliedsnummer`, `Unser Zeichen`, `Auftragsnummer`, `Kundennummer` |
| Contract, policy | contract / policy number | `Vertragsnummer`, `Versicherungsschein-Nr.`, `Police` |
| Certificate, report | report / certificate number | `Bericht-Nr.`, `Zertifikat-Nr.`, `Report No`, `No.`, `Nr.` (No. and Nr. only in the document's header area) |
| Enclosure without letterhead | empty unless labelled as its own number | `Unser Zeichen` |

Letters from a court, an arbitration or ombudsman body or a public authority:
its case / proceedings number is the sender's own reference. Labels:
`Verfahren`, `…verfahren <nr>`, `Az.`, `Aktenzeichen`.

Never a doc_number: headings (`RECHNUNG RK7`), page numbers (`1/2`),
form codes in margins, ZIP codes, account/card/customer numbers on
statements, third parties' case numbers when the sender's own reference
is printed, transaction IDs.

## 5. Amounts (invoices, receipts, credit notes)

- `total_gross` = grand total (BT-112), `total_net` = sum net (BT-109),
  `tax_amount` = total VAT (BT-110), `due_payable` = amount due (BT-115).
- Credit notes keep the printed amounts; the document type carries the sign.
- Subtotals, partial sums (e.g. labour only) and carry-over rows are not totals.
- Bank transaction confirmations (§2.8): the transfer amount is no invoice
  total; `total_gross`, `total_net` and `tax_amount` stay empty and the amount
  is the transaction amount.

## 6. due_date

- Invoice: printed payment due date (`zahlbar bis`, `fällig am`), or the
  printed direct-debit date (`buchen wir am`, `Abbuchung am`).
- Letter: only a deadline the recipient must meet by a printed date.
- Statement: only if a debit date for the balance is printed
  (`wird am ... belastet`, `fällig am`).

## 7. IBAN

- Invoice: the payee account (BT-84). A printed direct-debit account of the
  buyer is the payer account (BT-91), a separate field. Several printed
  payee accounts are all kept; the first in reading order is the primary one.
- Credit note: the account the credit is paid to, if printed; else empty.
- Statement: the account's own IBAN.
- Letter: the payee account only if a payment is requested.
- An IBAN that fails mod-97 is never output.
- Bank identifiers (BLZ, account number, BIC) are never a tax ID, VAT ID or
  IBAN unless they pass that field's format check.
- §2.8: the ordering customer (`Auftraggeber`, `Kontoinhaber`) is never the
  sender.

## 8. Statements

- `period_start` / `period_end`: only if a period is printed (`vom ... bis`,
  `Zeitraum`). Without a printed period, `period_end` = statement date and
  `period_start` = the day after the printed previous statement date, else empty.
- Balances: as printed; debit balances and credit-card debt are negative;
  opening balance, closing balance and transactions use one convention.
- Transactions: every booking row of all pages in printed order; carry-over,
  subtotal and header rows are not transactions.
