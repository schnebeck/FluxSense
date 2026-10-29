# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/stages/identity.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Identity heuristics ported from KPaperFlux core/validators.py
                (fuzzy ZIP/name matching, analytical validation of the
                Stage 1 tenant context / direction) plus party-based
                direction detection used for ZUGFeRD documents.
------------------------------------------------------------------------------
"""

import difflib
import re
from typing import Any, Dict, List, Optional, Tuple

from fluxsense.catalog.contract import is_address_only, is_document_title, not_a_party
from fluxsense.grounding.normalize import compact
from fluxsense.models.identity import IdentityProfile

BUSINESS_KEYWORDS: Tuple[str, ...] = ("gmbh", "ag", "gbr", "kg", "inc", "ltd", "corp", "ust-idnr", "vat id", "rechnung", "invoice")
BUSINESS_SUFFIXES: Tuple[str, ...] = ("gmbh", "ag", "gbr", "kg")


def check_identity_fuzzy(ocr_text: str, profile: Optional[IdentityProfile]) -> bool:
    """
    Validates if an identity profile is likely present in a text (KPaperFlux heuristic):
    1. ZIP code match, 2. fuzzy word matching, 3. prefix search for long names.

    Args:
        ocr_text: Page text.
        profile: Profile to search for.

    Returns:
        True if the identity is found with sufficient confidence.
    """
    if not profile or profile.is_empty():
        return False
    text_lower = ocr_text.lower()
    for keyword in profile.address_keywords:
        if keyword.isdigit() and len(keyword) == 5 and keyword in text_lower:
            return True
    for iban in profile.iban:
        if iban and compact(iban) in compact(ocr_text):
            return True
    terms: List[str] = []
    if profile.company_name:
        terms.append(profile.company_name)
    terms += [k for k in profile.address_keywords if not k.isdigit()]
    if profile.name:
        terms.append(profile.name)
    terms += profile.aliases
    words = text_lower.split()
    for term in terms:
        clean = term.lower()
        if not clean:
            continue
        if difflib.get_close_matches(clean, words, n=1, cutoff=0.6):
            return True
        if len(clean) > 5 and clean[:4] in text_lower:
            return True
    return False


def strong_identity_match(text: str, profile: Optional[IdentityProfile]) -> bool:
    """
    Strict identity match for party decisions (direction, form semantics):
    IBAN or VAT ID, or the full (normalized) name/company/alias, or a ZIP
    code together with a name token.

    Args:
        text: Party text.
        profile: Identity profile.

    Returns:
        True on a reliable match.
    """
    if not profile or profile.is_empty() or not text:
        return False
    from fluxsense.grounding.normalize import normalize_text
    flat, norm = compact(text), f" {normalize_text(text)} "
    if any(i and compact(i) in flat for i in profile.iban) or (profile.vat_id and compact(profile.vat_id) in flat):
        return True
    names = [n for n in [profile.name, profile.company_name] + profile.aliases + profile.company_aliases if n]
    if any(f" {normalize_text(n)} " in norm for n in names if len(normalize_text(n)) >= 5):
        return True
    zips = [k for k in profile.address_keywords if k.isdigit() and len(k) == 5]
    tokens = {t for n in names for t in normalize_text(n).split() if len(t) >= 4}
    return any(z in text for z in zips) and any(f" {t} " in norm for t in tokens)


def validate_structure(response: Dict[str, Any], pages: List[str], private_id: Optional[IdentityProfile],
                       business_id: Optional[IdentityProfile]) -> List[str]:
    """
    Analytical validation of Stage 1 tenant context / direction (KPaperFlux
    validate_ai_structure_response).

    Args:
        response: Stage 1.1 JSON.
        pages: Text per page (index 0 = page 1).
        private_id: Private profile.
        business_id: Business profile.

    Returns:
        Error strings (CONTEXT_ERROR / DIRECTION_ERROR mention 'Page N').
    """
    errors: List[str] = []
    entities = response.get("detected_entities", [])
    if not isinstance(entities, list):
        return errors
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        ctx = ent.get("tenant_context")
        ent_pages = ent.get("page_indices") or []
        if not ent_pages:
            continue
        try:
            first_text = pages[int(ent_pages[0]) - 1]
        except (ValueError, TypeError, IndexError):
            continue
        is_business = ctx == "BUSINESS"
        is_private = ctx == "PRIVATE"
        target = business_id if is_business else private_id if is_private else None
        if target is not None and target.is_empty():
            target = None
        match = bool(target) and check_identity_fuzzy(first_text, target)
        if not match:
            low = first_text.lower()
            if is_business and any(k in low for k in BUSINESS_KEYWORDS):
                match = True
            elif is_private and not any(k in low for k in BUSINESS_SUFFIXES):
                match = True
        if not match and float(ent.get("confidence") or 0.0) >= 0.98 and len(first_text.strip()) > 50:
            match = True
        if not match and target:
            errors.append(f"CONTEXT_ERROR: AI detected {ctx}, but no ZIP code or fuzzy address match was found on "
                          f"Page {ent_pages[0]}. Verification of Billing Address failed.")
        direction = ent.get("direction")
        if direction in ("INBOUND", "OUTBOUND") and target and not match:
            errors.append(f"DIRECTION_ERROR: AI detected {direction}, but user identity role could not be analytically "
                          f"verified on Page {ent_pages[0]}.")
    return errors


def party_text(party: Optional[Dict[str, Any]]) -> str:
    """Flattens a party dict into one text for matching."""
    if not party:
        return ""
    return " ".join(str(v) for v in party.values() if v)


def business_only(business_id: Optional[IdentityProfile], private_id: Optional[IdentityProfile]) -> Optional[IdentityProfile]:
    """
    The business profile without the person names that are also the
    private identity (a business profile often lists its owner's name as an
    alias). BUSINESS then needs a company name / alias, VAT ID, IBAN or the
    business address; a person-only match falls to the private profile.
    """
    if not business_id or business_id.is_empty():
        return business_id
    from fluxsense.grounding.normalize import normalize_text
    private_names = {normalize_text(n) for n in ([private_id.name] + private_id.aliases if private_id else []) if n}

    def personal(name: Optional[str]) -> bool:
        return bool(name) and normalize_text(name) in private_names

    return business_id.model_copy(update={
        "name": None if personal(business_id.name) else business_id.name,
        "aliases": [a for a in business_id.aliases if not personal(a)],
    })


def direction_from_parties(sender: Optional[Dict[str, Any]], recipient: Optional[Dict[str, Any]],
                           private_id: Optional[IdentityProfile], business_id: Optional[IdentityProfile]) -> Tuple[str, str]:
    """
    Derives direction and tenant context from sender/recipient party data.

    Args:
        sender: Sender party dict.
        recipient: Recipient party dict.
        private_id: Private profile.
        business_id: Business profile.

    Returns:
        Tuple (direction, tenant_context).
    """
    sender_txt, recipient_txt = party_text(sender), party_text(recipient)
    business = business_only(business_id, private_id)
    for context, profile in (("BUSINESS", business), ("PRIVATE", private_id)):
        if not profile or profile.is_empty():
            continue
        in_sender = bool(sender_txt) and strong_identity_match(sender_txt, profile)
        in_recipient = bool(recipient_txt) and strong_identity_match(recipient_txt, profile)
        if in_sender and in_recipient:
            return "INTERNAL", context
        if in_recipient:
            return "INBOUND", context
        if in_sender:
            return "OUTBOUND", context
    return "UNKNOWN", "UNKNOWN"


COMPANY_MARKERS = re.compile(r"\b(gmbh|ag|kg|ohg|se|eg|e\.v\.|ug|mbh|inc|ltd|llc|versicherung|bank|sparkasse|stadtwerke|"
                             r"amt|behörde|finanzamt|krankenkasse)\b", re.IGNORECASE)


def looks_like_company(party: Optional[Dict[str, Any]]) -> bool:
    """True if a party is (most likely) an organisation."""
    if not party:
        return False
    if party.get("company"):
        return True
    return bool(COMPANY_MARKERS.search(party_text(party)))


def apply_form_semantics(meta: Dict[str, Any], private_id: Optional[IdentityProfile],
                         business_id: Optional[IdentityProfile]) -> Tuple[bool, str]:
    """
    For forms filled in by the user: the filler is the sender, the form issuer
    the recipient, direction OUTBOUND. Swaps sender/recipient in place if the
    model took the printed issuer as sender.

    Args:
        meta: meta_header dict (modified in place).
        private_id: Private identity.
        business_id: Business identity.

    Returns:
        Tuple (swapped, reason).
    """
    sender, recipient = meta.get("sender") or {}, meta.get("recipient") or {}
    profiles = [p for p in (private_id, business_id) if p and not p.is_empty()]
    if profiles:
        user_sender = any(strong_identity_match(party_text(sender), p) for p in profiles) if sender else False
        user_recipient = any(strong_identity_match(party_text(recipient), p) for p in profiles) if recipient else False
        if user_recipient and not user_sender:
            meta["sender"], meta["recipient"] = recipient, sender
            return True, "identity profile found in the recipient block of a form"
        if user_sender:
            return False, "identity already in sender block"
    # Any person filling in a form (also a relative, not only the user) is the sender; the issuer is the recipient.
    if looks_like_company(sender) and recipient and not looks_like_company(recipient):
        meta["sender"], meta["recipient"] = recipient, sender
        return True, "printed organisation taken as sender of a form filled in by a person"
    return False, "no swap needed"


def unfilled_form(audit: Optional[Dict[str, Any]]) -> bool:
    """
    Field contract §2.5: a pre-addressed cover or return sheet that is not
    filled in - the full visual audit ran and found neither a signature nor
    handwriting. Its parties follow the letter rule (issuer = sender,
    printed address = recipient) instead of the form rule (§2.4).
    """
    if not isinstance(audit, dict) or not isinstance(audit.get("signatures"), dict):
        return False
    handwriting = bool((audit.get("audit_summary") or {}).get("has_handwriting"))
    return not audit["signatures"].get("has_signature") and not handwriting


STATEMENT_TYPES = frozenset({"BANK_STATEMENT", "CREDIT_CARD_STATEMENT"})


def dedupe_party(party: Optional[Dict[str, Any]]) -> bool:
    """Drops a 'name' that repeats the 'company' (e.g. 'Musterbank' + 'Musterbank'). Returns True if changed."""
    if not isinstance(party, dict) or not party.get("name") or not party.get("company"):
        return False
    from fluxsense.grounding.normalize import normalize_text
    name, company = normalize_text(party["name"]), normalize_text(party["company"])
    if name == company or name in company:
        party.pop("name")
        return True
    return False


def harmonize_parties(meta: Dict[str, Any], direction: str, type_tags: List[str], private_id: Optional[IdentityProfile],
                      business_id: Optional[IdentityProfile]) -> Tuple[List[str], str]:
    """
    Makes sender/recipient consistent with the direction and the document type:
    statements: sender = bank, recipient = account holder (swap if needed);
    other documents: the layout roles are trusted - with the user's identity in
    the sender block (and not in the recipient block) the direction becomes
    OUTBOUND, with the identity only in the recipient block INBOUND; duplicate
    name/company tokens are removed.

    Args:
        meta: meta_header (modified in place).
        direction: Document direction.
        type_tags: Type tags (FluxSense vocabulary).
        private_id: Private identity.
        business_id: Business identity.

    Returns:
        Tuple (notes describing the changes, resulting direction).
    """
    notes: List[str] = []
    for key in ("sender", "recipient"):
        if dedupe_party(meta.get(key)):
            notes.append(f"duplicate name removed in {key}")
    sender, recipient = meta.get("sender") or {}, meta.get("recipient") or {}
    profiles = [p for p in (private_id, business_id) if p and not p.is_empty()]
    user_sender = any(strong_identity_match(party_text(sender), p) for p in profiles) if sender else False
    user_recipient = any(strong_identity_match(party_text(recipient), p) for p in profiles) if recipient else False
    swap = False
    if STATEMENT_TYPES & set(type_tags):
        if (user_sender and not user_recipient) or (not looks_like_company(sender) and looks_like_company(recipient)):
            swap, reason = True, "statement: sender must be the bank, recipient the account holder"
    elif user_sender and not user_recipient and direction != "OUTBOUND":
        notes.append(f"direction {direction} -> OUTBOUND (the user's identity is in the sender block)")
        direction = "OUTBOUND"
    elif user_recipient and not user_sender and direction == "OUTBOUND":
        notes.append("direction OUTBOUND -> INBOUND (the user's identity is in the recipient block)")
        direction = "INBOUND"
    if swap and (sender or recipient):
        meta["sender"], meta["recipient"] = recipient, sender
        notes.append(f"sender/recipient swapped ({reason})")
    return notes, direction


ROLE_WORDS = frozenset({
    "patient", "patientin", "patienten", "versicherte", "versicherter", "versicherten", "selbstzahler", "privat", "private",
    "antragsteller", "antragstellerin", "kunde", "kundin", "mitglied", "versicherungsnehmer", "versicherungsnehmerin",
    "arbeitgeber", "arbeitnehmer", "empfanger", "empfaenger", "absender", "beihilfeberechtigte", "beihilfeberechtigter",
    "kontoinhaber", "kontoinhaberin", "inhaber", "inhaberin", "vertreter", "erziehungsberechtigte", "unterschrift",
    "name", "vorname", "anschrift", "adresse", "fur", "fuer", "und", "oder", "bzw",
})


def is_role_label(value: Optional[str]) -> bool:
    """True for form role labels/headings such as 'Patient/ Patientin' or 'Privat Versicherte/ Selbstzahler'."""
    from fluxsense.grounding.normalize import normalize_text
    tokens = [t for t in normalize_text(value or "").split() if len(t) >= 2]
    return bool(tokens) and all(t in ROLE_WORDS for t in tokens)


def remove_role_labels(meta: Dict[str, Any]) -> List[str]:
    """Removes form role labels/headings, info-column labels and postal service marks taken as sender/recipient names."""
    notes: List[str] = []
    for key in ("sender", "recipient"):
        party = meta.get(key)
        if not isinstance(party, dict):
            continue
        for sub in ("company", "name"):
            if party.get(sub) and is_role_label(str(party[sub])):
                notes.append(f"{key}.{sub} {party.pop(sub)!r} removed (form role label, not a party)")
            elif party.get(sub) and not_a_party(party[sub]):
                notes.append(f"{key}.{sub} {party.pop(sub)!r} removed (info-column label / postal service mark, not a party)")
            elif party.get(sub) and is_document_title(party[sub]):
                notes.append(f"{key}.{sub} {party.pop(sub)!r} removed (document heading / title, not a party)")
            elif party.get(sub) and is_address_only(party[sub]):
                notes.append(f"{key}.{sub} {party.pop(sub)!r} removed (street / P.O. box / city only, not a party name)")
    return notes
