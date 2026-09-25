"""The perception layer (Phase 4): everything JARVIS takes in besides the words typed to it.

Text stays the primary way the user talks to JARVIS. Images, screenshots, documents and (when the user enables it)
the screen become *observations*: recorded inputs with a kind, a source, a time, a session, provenance and a
sensitivity, stored once (content-addressed) and analysed on demand. What is learned from an observation — a
description, OCR text, UI elements, a document's structure — is derived data attached to it, cached, and kept
separate from instructions: text inside an image or document is something JARVIS read, never something it was told
to do.

Modules:
    inputs        the input abstraction (kinds, sources, observations, normalisation; voice and camera are
                  interfaces only, for later)
    images        image formats, validation and preparation for a model (no required dependencies)
    store         observations, their stored copies, derived results, retention and forgetting
    safety        untrusted-content framing, instruction-like text, sensitive-content detection
    vision        analysis through a vision-capable model (routed; never pretends)
    ocr           reading text from images, with confidence and provenance
    documents     reading documents and files without loading them whole into a model
    screen        screen awareness: off unless the user turns it on
    devices       discovering input devices (never switching them on)
    capabilities  what JARVIS can currently perceive, stated accurately
    references    "this", "the previous one", "the second screenshot", "the diagram from yesterday"
    ui            UI elements and the (future) UI action interface
    service       the facade the rest of JARVIS uses
    tools         perception tools for the model, planner and agents
"""
