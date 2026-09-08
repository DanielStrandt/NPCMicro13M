"""Double-clickable Tkinter conversation GUI for the NPCMicro13M bundle."""

from __future__ import annotations

import io
import queue
import sys
import threading
from contextlib import redirect_stdout
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uomind_api import UOMindRuntime  # noqa: E402


class UOMindGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("NPCMicro13M — NPC Conversation")
        self.root.geometry("1000x760")
        self.root.minsize(780, 600)

        self.runtime: UOMindRuntime | None = None
        self.state = ""
        self.busy = False
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()

        self.name_var = tk.StringVar()
        self.role_var = tk.StringVar()
        self.home_var = tk.StringVar()
        self.raw_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Fill in the NPC, then press Start Conversation.")

        self._build_style()
        self._build_layout()
        self.root.after(100, self._poll_events)

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Status.TLabel", foreground="#555555")

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)

        ttk.Label(outer, text="NPCMicro13M NPC Conversation", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            outer,
            text="Define the NPC once. Chat history is display-only; every reply is a single-pass turn.",
            style="Status.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 10))

        setup = ttk.LabelFrame(outer, text="NPC details", padding=10)
        setup.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        setup.columnconfigure(1, weight=1)
        setup.columnconfigure(3, weight=1)

        ttk.Label(setup, text="Name").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
        self.name_entry = ttk.Entry(setup, textvariable=self.name_var)
        self.name_entry.grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(setup, text="Profession / role").grid(row=0, column=2, sticky="w", padx=(18, 6), pady=4)
        self.role_entry = ttk.Entry(setup, textvariable=self.role_var)
        self.role_entry.grid(row=0, column=3, sticky="ew", pady=4)

        ttk.Label(setup, text="Town / area").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=4)
        self.home_entry = ttk.Entry(setup, textvariable=self.home_var)
        self.home_entry.grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Label(setup, text="Additional facts").grid(row=1, column=2, sticky="nw", padx=(18, 6), pady=4)
        self.facts_text = tk.Text(setup, height=3, width=40, wrap="word", relief="solid", borderwidth=1)
        self.facts_text.grid(row=1, column=3, sticky="ew", pady=4)

        controls = ttk.Frame(setup)
        controls.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        self.start_button = ttk.Button(controls, text="Start Conversation", command=self.start_conversation)
        self.start_button.pack(side="left")
        self.edit_button = ttk.Button(controls, text="Edit NPC", command=self.edit_npc, state="disabled")
        self.edit_button.pack(side="left", padx=(8, 0))
        self.clear_button = ttk.Button(controls, text="Clear Chat", command=self.clear_chat)
        self.clear_button.pack(side="left", padx=(8, 0))
        ttk.Checkbutton(
            controls,
            text="Use raw model output (bypass grounding)",
            variable=self.raw_var,
        ).pack(side="right")

        chat_frame = ttk.LabelFrame(outer, text="Conversation", padding=8)
        chat_frame.grid(row=3, column=0, sticky="nsew")
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        self.chat = scrolledtext.ScrolledText(
            chat_frame,
            wrap="word",
            state="disabled",
            font=("Segoe UI", 10),
            padx=8,
            pady=8,
        )
        self.chat.grid(row=0, column=0, sticky="nsew")
        self.chat.tag_configure("npc", foreground="#174a7c", spacing1=5)
        self.chat.tag_configure("player", foreground="#333333", spacing1=5)
        self.chat.tag_configure("system", foreground="#777777", spacing1=5)

        input_frame = ttk.Frame(chat_frame)
        input_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        input_frame.columnconfigure(0, weight=1)
        self.player_entry = ttk.Entry(input_frame, state="disabled")
        self.player_entry.grid(row=0, column=0, sticky="ew")
        self.player_entry.bind("<Return>", self._send_from_event)
        self.player_entry.bind("<Control-Return>", self._send_from_event)
        self.send_button = ttk.Button(input_frame, text="Send", command=self.send_message, state="disabled")
        self.send_button.grid(row=0, column=1, padx=(8, 0))

        ttk.Label(outer, textvariable=self.status_var, style="Status.TLabel").grid(
            row=4, column=0, sticky="w", pady=(8, 0)
        )

    @staticmethod
    def _article_for(role: str) -> str:
        role = role.strip()
        if role.lower().startswith(("a ", "an ", "the ")):
            return role
        return ("an " if role[:1].lower() in "aeiou" else "a ") + role

    def _build_state(self) -> str:
        name = self.name_var.get().strip()
        role = self.role_var.get().strip()
        home = self.home_var.get().strip()
        facts = " ".join(self.facts_text.get("1.0", "end").split())
        parts = []
        if name:
            parts.append(f"Your name is {name}.")
        if role and home:
            parts.append(f"You are {self._article_for(role)} from {home}.")
        elif role:
            parts.append(f"You are {self._article_for(role)}.")
        elif home:
            parts.append(f"You are from {home}.")
        if facts:
            parts.append(facts if facts.endswith(('.', '!', '?')) else facts + ".")
        return " ".join(parts)

    def start_conversation(self) -> None:
        if self.busy:
            return
        state = self._build_state()
        if not state:
            messagebox.showwarning("NPC details needed", "Enter at least a name, profession, town, or fact.")
            return
        self.state = state
        self.busy = True
        self.start_button.configure(state="disabled")
        for entry in (self.name_entry, self.role_entry, self.home_entry):
            entry.configure(state="disabled")
        self.facts_text.configure(state="disabled")
        self.status_var.set("Loading the model… this may take a few seconds the first time.")
        self._append("SYSTEM", f"NPC state locked: {self.state}", "system")
        threading.Thread(target=self._load_runtime, daemon=True).start()

    def _load_runtime(self) -> None:
        try:
            captured = io.StringIO()
            with redirect_stdout(captured):
                runtime = UOMindRuntime(bundle=ROOT, device="auto", precision="auto")
            self.events.put(("loaded", runtime))
        except Exception as exc:  # surfaced on the GUI thread
            self.events.put(("error", exc))

    def edit_npc(self) -> None:
        if self.busy:
            return
        self.start_button.configure(state="normal")
        self.edit_button.configure(state="disabled")
        for entry in (self.name_entry, self.role_entry, self.home_entry):
            entry.configure(state="normal")
        self.facts_text.configure(state="normal")
        self.status_var.set("Edit the NPC, then press Start Conversation.")

    def clear_chat(self) -> None:
        self._set_chat("")
        if self.state:
            self._append("SYSTEM", f"NPC state: {self.state}", "system")

    def _send_from_event(self, _event: object) -> str:
        self.send_message()
        return "break"

    def send_message(self) -> None:
        if self.runtime is None or self.busy:
            return
        player = self.player_entry.get().strip()
        if not player:
            return
        self.player_entry.delete(0, "end")
        self._append("YOU", player, "player")
        self._append("NPC", "Thinking…", "npc")
        self.busy = True
        self.send_button.configure(state="disabled")
        self.status_var.set("Generating response…")
        threading.Thread(target=self._generate, args=(self.state, player, self.raw_var.get()), daemon=True).start()

    def _generate(self, state: str, player: str, raw_model: bool) -> None:
        # Deliberately pass only the fixed NPC state and current player text.
        # The visible chat transcript is never included in model inference.
        try:
            result = self.runtime.respond(state, player, raw_model=raw_model)  # type: ignore[union-attr]
            self.events.put(("answer", result))
        except Exception as exc:
            self.events.put(("error", exc))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "loaded":
                    self.runtime = value  # type: ignore[assignment]
                    self.busy = False
                    self.edit_button.configure(state="normal")
                    self.player_entry.configure(state="normal")
                    self.send_button.configure(state="normal")
                    self.player_entry.focus_set()
                    self.status_var.set("Ready. Type a player message and press Enter or Send.")
                elif kind == "answer":
                    result = value  # type: ignore[assignment]
                    self._replace_thinking(result["text"])
                    self.busy = False
                    self.send_button.configure(state="normal")
                    self.status_var.set(
                        "Grounded response applied." if result["grounded_applied"] else "Raw model answer retained."
                    )
                    self.player_entry.focus_set()
                elif kind == "error":
                    self.busy = False
                    self.start_button.configure(state="normal")
                    self.send_button.configure(state="normal" if self.runtime else "disabled")
                    self.status_var.set("Something went wrong. See the error message.")
                messagebox.showerror("NPCMicro13M error", str(value))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _append(self, speaker: str, text: str, tag: str) -> None:
        self.chat.configure(state="normal")
        self.chat.insert("end", f"{speaker}: {text}\n\n", tag)
        self.chat.see("end")
        self.chat.configure(state="disabled")

    def _set_chat(self, text: str) -> None:
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        if text:
            self.chat.insert("end", text)
        self.chat.configure(state="disabled")

    def _replace_thinking(self, answer: str) -> None:
        self.chat.configure(state="normal")
        content = self.chat.get("1.0", "end-1c")
        marker = "NPC: Thinking…\n\n"
        index = content.rfind(marker)
        if index >= 0:
            self.chat.delete(f"1.0 + {index} chars", "end")
            self.chat.insert("end", f"NPC: {answer}\n\n", "npc")
        else:
            self.chat.insert("end", f"NPC: {answer}\n\n", "npc")
        self.chat.see("end")
        self.chat.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    UOMindGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
