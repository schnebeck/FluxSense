# FluxSense Field Contract

Version 1.6 (2026-10-30). This contract defines what each core field of a
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

**2.9 Signatures.** A name read only from a signature or a handwritten
signature line (`Unterschrift`, `Datum/Unterschrift Mieter`, `gez.`) is no
party: it becomes sender or recipient only if the same name is printed or
written elsewhere as a party. A reading that is not in the text at all and
points to a signature (its snippet is a signature line, or it carries the
other party's surname) is removed; a sender of a letter without a printed
sender stays empty. Forms filled in by a person (§2.4) are exempt: their
signer is the sender.

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
form codes in margins, form print numbers (a dotted digit code that is no
date on one of the last lines of a page, e.g. 32.9.021, or a value after
Formular / Vordruck / Form), ZIP codes, account/card/customer numbers on
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
- An IBAN that fails mod-97 is never output - unless the document prints
  it exactly so (character for character in the native text, or confirmed
  as printed on the zoom crop): then it is kept as printed, flagged with an
  error ("as printed fails the checksum; verify with the issuer"), the
  bank-code status and up to two one-digit corrections that pass mod-97
  and keep a known bank code (hints only, never applied), and never
  auto-accepted. A misread IBAN is removed.
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

## 9. Date consistency

The dates of one logical document must fit together. The check compares
`doc_date` (else the invoice or statement date) with the document's other
dates; `other - doc_date` must lie in the window (days):

| Other date | Window | Rule |
|---|---|---|
| `due_date` | -31 .. +366 | due at most a month before and a year after the invoice date |
| `order_date` | -731 .. +92 | ordered at most 2 years before the invoice date |
| `service_date` | -731 .. +366 | service / delivery within 2 years before and 1 year after |
| `service_period_start` / `_end` | -1096 .. +366 / +731 | service period around the document date |
| `effective_date` (contract) | open .. +731 | signed at most 2 years before the start; amendments and other later documents may follow at any time |
| `period_start` (statement) | -500 .. +31 | issued at most about 16 months after the period starts |
| `period_end` (statement, also against `statement_date`) | -366 .. +31 | issued within a year after the period ends |
| `payment_due_date` (statement) | -31 .. +366 | debit within a year after the statement date |
| `deadline` (letter) | -31 .. +731 | reply deadline within 2 years after the letter date |
| `pay_period_end` (payslip) | -366 .. +62 | payslip within a year after its pay period |

- A violation is a warning (`dates.consistency`) on both fields with a
  review reason naming the other field and its value. It is never fixed
  silently, and neither field is auto-accepted.
- **Ambiguous years.** A two-digit year that is not grounded in the text,
  and any zoom / vision reading, is ambiguous (one confused digit such as
  0/2, 6/8, 1/7, or the other century). Its candidate readings are the first
  model value, the current value, the dates in its snippets and their year
  variants. If exactly one candidate is consistent with the other date -
  which must be grounded in the text and not itself a visual reading - and
  not in the future, it replaces the value, which stays flagged
  (`dates.year_ambiguous`: "year ambiguous, chosen by consistency with ...").
  With no or several consistent candidates nothing is chosen and the field
  is flagged.
- A zoom-verified value that still conflicts loses its visual support (it
  counts as ungrounded).

## 10. Contracts: payment terms and duration

Contracts with recurring payments (rent, lease, subscriptions, loans) and
insurance policies carry their payment terms in `legal_body`:

- `payment_terms`: `amount` = the printed TOTAL recurring amount
  (`monatlich insgesamt zu zahlende Miete`), `currency`, `frequency`
  (MONTHLY, QUARTERLY, HALF_YEARLY, YEARLY, ONCE), `due_rule` as printed,
  `components` = every printed component (`label`, `amount`),
  `payment_account` (holder as `name`, `iban`, `bic`, `bank_name`) and
  `payment_reference` (`Verwendungszweck`).
- `deposit`: the security deposit (`Kaution`).
- `contract_object`: the object of the contract - a rented property
  (`street`, `house_number`, `zip_code`, `city`, `location_detail` such as
  `EG links`), a vehicle or meter (`identifier`).
- Check `contract.payment_sum`: two or more printed components must add
  up to the printed total (tolerance 0.01); when they do, the total and the
  components have provenance `check`. The payee IBAN gets the usual mod-97
  check and grounding.
- `minimum_term` (`value`, `unit`, `mutual_waiver`, `original_text`): a
  minimum term or a (mutual) waiver of ordinary termination for a duration
  (`verzichten wechselseitig auf ihr Recht zur Kündigung für die Dauer von 3
  Jahren`, `Mindestlaufzeit`). It is never the `notice_period`, which holds
  only a printed period of notice; `... erstmalig nach Ablauf ... mit
  gesetzlicher Frist` leaves `notice_period` empty. A waiver the model put
  into `notice_period` is moved to `minimum_term`.

## 11. Certificates and confirmations

- A form confirming a fact about a person and issued by a third party
  (landlord's `Wohnungsgeberbestätigung`, employer's `Bescheinigung`) is a
  CERTIFICATE, also when it is filled in by hand on a printed form; a
  LETTER / OTHER / APPLICATION answer with such a title is re-typed.
- Sender = the issuer (landlord, employer, school, authority); the
  issuer's name in a form field counts as printed.
- `legal_body`: `certified_persons`, `certified_event` (MOVE_IN, MOVE_OUT,
  EMPLOYMENT, MEMBERSHIP, COMPLETION, OTHER), `event_date` /
  `event_end_date`, `subject_address` (with `location_detail`), as printed or
  written.

## 12. Form fields read by their label

On scanned pages of forms (no native text layer, at least three printed
field labels of the lexicon - or only date / signature lines such as
`..., den`, `Ort, Datum`, `Datum/Unterschrift`, which are then read alone)
each printed label is read on its own:

- The label (`Ort, Datum`, `..., den`, `Einzug am`, `Name des ...`,
  `Anschrift ...`, `Geschoss / Lage`, `Telefon`, `für folgende Person(en)`,
  ...) gives the field kind (date, place and date, name, persons, address,
  location, phone).
- Its entry region is cropped from the scan's native resolution and read
  by the vision model with the label and a type hint only; the full-page
  value is never shown. Regions are clipped at neighbouring labels: an
  inline label (ending in `am`, `den`, `vom`, `bis`, `:` with ink to its
  right) uses only its own text band up to the next label on the line; a
  label printed under its entry line looks upward to the next label above;
  a header label (`... lauten:`) looks down to the next label below. An
  empty field is answered `EMPTY` and yields no value.
- A checkbox before the label decides: an unticked option is not read, the
  ticked one gives the certified event (`Einzug am` -> MOVE_IN); an unknown
  tick state is flagged (`fieldread.checkbox_unknown`).
- Dates are normalised to ISO. When separators and digits can be confused
  (`1|10|26`: a bar is a separator or a one) every valid reading is kept;
  more than one reading flags the field (`fieldread.ambiguous`, naming the
  readings).
- The readings form a FORM FIELDS block of the page text for the
  extraction and a `fieldread` text source for grounding; a value equal to
  a reading has provenance `label`, but a handwritten reading alone is
  never auto-accepted ("handwritten form field (single reading)"): it needs
  an independent second reading with the same value in another text source
  of the page (native text, OCR / layout text) - the full-page extraction
  saw the FORM FIELDS block and does not count - and, for dates, a single
  reading.
- Date readings include the `1` / `|` / `/` swaps of handwriting
  (`11/10/26` may be `1|10|26`); a name + address block is split into name
  and address parts (a name never becomes a street) or left as context.
- Unambiguous labels map directly: `Ort, Datum` / `..., den` -> doc_date,
  `Einzug am` / `Auszug am` -> certified_event and event_date, the persons
  -> certified_persons, the issuer's name and address -> sender, the flat
  address and floor -> subject_address. A differing full-page value is
  replaced by the reading and flagged with both values
  (`fieldread.disagreement`); a full-page value for an empty field is
  flagged.
