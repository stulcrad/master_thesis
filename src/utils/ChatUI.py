import ipywidgets as widgets
from IPython.display import display

from typing import Optional, List, Dict, Any
from pathlib import Path
import json, time

class ChatUI:
    """
    Interactive chat UI created using ipywidgets for Jupyter Notebooks
    for an OpenAI-compatible client (e.g., Ollama).

    Parameters
    ----------
    - client: OpenAI-compatible client instance.
    - system_prompt: Initial system prompt for the assistant.
    - model: Default model to use (if None, uses the first available model).
    - stream: Whether to stream responses or get them in one go.
    - chats_dir: Directory to save/load chat transcripts.
    """
    def __init__(self, client, *, 
                 system_prompt:str = "You are a helpful assistant.",
                 model: Optional[str] = None,
                 stream: bool = True,
                 chats_dir: Optional[str | Path] = None,
                 custom_input: Optional[str] = None) -> None:
        self.client = client
        self.system_prompt = system_prompt
        self.stream = stream

        # Fetch available models
        try:
            self._models = [m.id for m in self.client.models.list()]
            self._models = sorted(self._models)
        except Exception:
            self._models = ['gpt-oss:20b']
        self._default_model = model or "gpt-oss:20b" if "gpt-oss:20b" in self._models else self._models[0]

        # Create UI components
        self.model_dd = widgets.Dropdown(options=self._models, value=self._default_model, description="Model:")
        self.stream_cb = widgets.Checkbox(value=self.stream, description="Stream")
        self.system_tb = widgets.Textarea(value=self.system_prompt.strip(), description="System:", layout=widgets.Layout(width='99.7%', height='190px'))
        self.transcript = widgets.Textarea(value="", description="Chat:", layout=widgets.Layout(width="99.7%", height="400px"), disabled=True)
        if custom_input:
            self.user_in = widgets.Textarea(value=custom_input, description="You:", layout=widgets.Layout(width="100%", height="100px"))
        else:
            self.user_in = widgets.Textarea(placeholder="Type a message and press Send", description="You:", layout=widgets.Layout(width="100%", height="100px"))
        self.send_btn = widgets.Button(description="Send", button_style="primary")
        self.new_btn = widgets.Button(description="New chat")
        self.log_out = widgets.Output(layout=widgets.Layout(border="1px solid #eee"))
        self.cancel_btn = widgets.Button(description="Cancel", button_style="warning")

        # Temperature and penalties
        self.temp_defb = widgets.Checkbox(value=False, description="Model's default temp")
        self.temp_sl = widgets.FloatSlider(value=0.2, min=0.0, max=2.0, step=0.1, description="Temp", readout_format='.1f')
        self.freq_pen_sl = widgets.FloatSlider(value=0.0, min=-2.0, max=2.0, step=0.1, description="Freq Pen", readout_format='.1f')
        self.freq_pen_defb = widgets.Checkbox(value=True, description="Model's default freq pen")
        self.pres_pen_sl = widgets.FloatSlider(value=0.0, min=-2.0, max=2.0, step=0.1, description="Pres Pen", readout_format='.1f')
        self.pres_pen_defb = widgets.Checkbox(value=True, description="Model's default pres pen")

        # Initialize conversation state
        self.conv_messages: List[Dict[str, Any]] = []
        self.conv_messages.append({"role": "system", "content": self.system_prompt.strip()})

        # Chats directory
        self.CHATS_DIR = chats_dir or Path("/home/stulcrad/master_thesis/chats")
        self.CHATS_DIR.mkdir(parents=True, exist_ok=True)

        # Save/load widgets
        self.fname_tb = widgets.Text(value=self._default_name(), description="Filename:")
        self.save_btn = widgets.Button(description="Save", button_style="success")
        self.files_dd = widgets.Dropdown(options=self._list_chat_names(), description="Files:")
        self.refresh_btn = widgets.Button(description="Refresh")
        self.load_btn = widgets.Button(description="Load")
        self.status_out = widgets.Output()

        # Wire events to widgets
        self.temp_defb.observe(self._temp_toogle, names='value')
        self.send_btn.on_click(self._send_message)
        self.new_btn.on_click(self._new_chat)
        self.save_btn.on_click(self._save_chat)
        self.refresh_btn.on_click(self._refresh_files)
        self.load_btn.on_click(self._load_chat)
        self.freq_pen_defb.observe(self._freq_pen_toogle, names='value')
        self.pres_pen_defb.observe(self._pres_pen_toogle, names='value')

        # Build layout
        controls1 = widgets.HBox([self.model_dd, self.stream_cb, self.temp_defb, self.freq_pen_defb, self.pres_pen_defb, self.new_btn])
        temp_pens_controls = widgets.HBox([self.temp_sl, self.freq_pen_sl, self.pres_pen_sl])
        controls2 = widgets.HBox([self.user_in, self.send_btn])
        controls_sv = widgets.HBox([self.fname_tb, self.save_btn])
        controls_ld = widgets.HBox([self.files_dd, self.refresh_btn, self.load_btn])
        self.ui = widgets.VBox([controls1, temp_pens_controls, self.system_tb, self.transcript, controls2, controls_sv, controls_ld, self.status_out, self.log_out])

        # Initial UI state
        self._render_transcript()
        self._update_temp_ui()
        self._update_freq_ui()
        self._update_pres_ui()
        display(self.ui)

    def _list_chat_names(self) -> List[str]:
        """ List available chat JSON files in the chats directory """
        files = sorted(self.CHATS_DIR.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        return [f.name for f in files]
    
    def _disable_ui(self, disabled: bool) -> None:
        """ Enable/disable UI widgets during processing """
        for w in (self.model_dd, self.stream_cb, self.temp_sl, self.temp_defb, self.user_in, self.send_btn, self.new_btn, self.system_tb,
                  self.fname_tb, self.save_btn, self.load_btn, self.refresh_btn):
            w.disabled = disabled

    def _now_stamp(self) -> str:
        """ Get current timestamp as a string for filenames """
        return time.strftime("%Y-%d|%m-%H:%M")
    
    def _default_name(self) -> str:
        """ Generate a default filename based on the current timestamp """
        return f"chat-{self._now_stamp()}.json"
    
    def _refresh_files(self, _=None, limit: Optional[int] = None) -> None:
        """ List available chat JSON files, sorted by modification time """
        names = self._list_chat_names()
        if limit is not None:
            names = names[:limit]
        self.files_dd.options = names

    def _update_temp_ui(self) -> None:
        """ Update temperature slider state based on checkbox """
        self.temp_sl.disabled = self.temp_defb.value

    def _temp_toogle(self, change) -> None:
        """ Update temperature slider state when checkbox changes """
        if change['name'] == 'value':
            self._update_temp_ui()

    def _update_freq_ui(self) -> None:
        """ Update frequency penalty slider state based on checkbox """
        self.freq_pen_sl.disabled = self.freq_pen_defb.value

    def _freq_pen_toogle(self, change) -> None:
        """ Update frequency penalty slider state when checkbox changes """
        if change['name'] == 'value':
            self._update_freq_ui()

    def _update_pres_ui(self) -> None:
        """ Update presence penalty slider state based on checkbox """
        self.pres_pen_sl.disabled = self.pres_pen_defb.value

    def _pres_pen_toogle(self, change) -> None:
        """ Update presence penalty slider state when checkbox changes """
        if change['name'] == 'value':
            self._update_pres_ui()

    def _render_transcript(self) -> None:
        """ Render the conversation messages into the transcript textarea """
        lines: List[str] = []
        for msg in self.conv_messages:
            role = msg.get("role", "?") # Get the role
            content = msg.get("content", "") # Get the content
            if role == "system":
                lines.append(f"[system] {content}")
            elif role == "user":
                lines.append(f"=== You ===:\n{content}")
            else:
                lines.append("=== Assistant ===:")
                r = msg.get("reasoning")
                if r:
                    lines.append(f"[Thinking]:\n{r}\n[/Thinking]\n")
                lines.append(content)
            lines.append("")
        # Join lines by newlines and set to show on the transcript
        self.transcript.value = "\n".join(lines).rstrip()

    def _send_message(self, _=None) -> None:
        """ Send user message, get assistant reply, update transcript """
        text = self.user_in.value.strip()
        if not text:
            return
        self._disable_ui(True)
        self.user_in.value = ""

        self.conv_messages.append({"role": "user", "content": text})
        self._render_transcript()

        model = self.model_dd.value

        if model.startswith("qwen"):
            self.conv_messages[0]['content'] += "\n\\no_think"

        try:
            if self.stream_cb.value:
                req_kwargs: Dict[str, Any] = dict(model=model, messages=self.conv_messages, stream=True)
                if not self.temp_defb.value: # Use custom temp
                    req_kwargs['temperature'] = float(self.temp_sl.value)
                if not self.freq_pen_defb.value:
                    req_kwargs['frequency_penalty'] = float(self.freq_pen_sl.value)
                if not self.pres_pen_defb.value:
                    req_kwargs['presence_penalty'] = float(self.pres_pen_sl.value)
                if model.startswith("gpt-oss"):
                    req_kwargs['reasoning_effort'] = 'low' # Request low reasoning from gpt-oss models

                reasoning_text = "" # Accumulate streamed reasoning here
                assistant_text = "" # Accumulate streamed text here
                stream_resp = self.client.chat.completions.create(**req_kwargs)
                # Add placeholder assistant turn
                self.conv_messages.append({"role": "assistant", "content": ""})
                for chunk in stream_resp:
                    try:        
                        delta = chunk.choices[0].delta
                        reasoning_piece = getattr(delta, 'reasoning', None) if delta else None
                        piece = getattr(delta, 'content', None) if delta else None
                    except Exception:
                        piece = None
                    if reasoning_piece:
                        reasoning_text += reasoning_piece
                        self.conv_messages[-1]['reasoning'] = reasoning_text
                        self._render_transcript()
                    if piece:
                        assistant_text += piece
                        self.conv_messages[-1]['content'] = assistant_text
                        self._render_transcript()
            else:
                req_kwargs = dict(model=model, messages=self.conv_messages)
                if not self.temp_defb.value:
                    req_kwargs['temperature'] = float(self.temp_sl.value)
                if not self.freq_pen_defb.value:
                    req_kwargs['frequency_penalty'] = float(self.freq_pen_sl.value)
                if not self.pres_pen_defb.value:
                    req_kwargs['presence_penalty'] = float(self.pres_pen_sl.value)
                if model.startswith("gpt-oss"):
                    req_kwargs['reasoning_effort'] = 'low'

                # Send request
                resp = self.client.chat.completions.create(**req_kwargs)
                # Get assistant text
                assistant_text = resp.choices[0].message.content 
                # Add assistant message
                self.conv_messages.append({"role": "assistant", "content": assistant_text})
                # Optional reasoning
                reasoning = getattr(resp.choices[0].message, 'reasoning', None) 
                if reasoning:
                    self.conv_messages[-1]['reasoning'] = reasoning
        except Exception as e:
            with self.log_out:
                print(f"Error: {e}")
        finally:
            self._disable_ui(False)
            self._render_transcript()
            self._update_temp_ui()
            self._update_freq_ui()
            self._update_pres_ui()

    def _new_chat(self, _=None) -> None:
        """ Start a new chat session """
        self.conv_messages.clear()
        sys_text = self.system_tb.value.strip()
        if sys_text:
            self.conv_messages.append({"role": "system", "content": sys_text})
        self.user_in.value = ""
        self._render_transcript()
    
    def _save_chat(self, _=None) -> None:
        """ Save current chat to a JSON file """
        name = (self.fname_tb.value or self._default_name()).strip() # Get filename
        try:
            # Create dictionary to save
            data = {
                "model": self.model_dd.value,
                "created": self._now_stamp(),
                "temperature": "default" if self.temp_defb.value else self.temp_sl.value,
                "frequency_penalty": "default" if self.freq_pen_defb.value else self.freq_pen_sl.value,
                "presence_penalty": "default" if self.pres_pen_defb.value else self.pres_pen_sl.value,
                "messages": self.conv_messages,
            }
            # Full path
            path = self.CHATS_DIR / name

            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            with self.status_out:
                print(f"Saved: {path}")

            self._refresh_files()
        except Exception as e:
            with self.log_out:
                print(f"Error saving chat: {e}")
    
    def _load_chat(self, _=None) -> None:
        """ Load a chat from the selected file """
        short_path = self.files_dd.value
        if not short_path:
            return
        try:
            file_path = self.CHATS_DIR / short_path
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            messages = data.get("messages", [])
            model = data.get("model")

            self.conv_messages.clear()
            self.conv_messages.extend(messages)

            if model and (model in getattr(self.model_dd, 'options', [])):
                self.model_dd.value = model

            if self.conv_messages and self.conv_messages[0].get("role") == "system":
                self.system_tb.value = self.conv_messages[0]["content"]

            if data.get("temperature") == "default":
                self.temp_defb.value = True
            else:
                self.temp_defb.value = False
                try:
                    self.temp_sl.value = float(data.get("temperature", 0.2))
                except Exception:
                    pass

            if data.get("frequency_penalty") == "default":
                self.freq_pen_defb.value = True
            else:
                self.freq_pen_defb.value = False
                try:
                    self.freq_pen_sl.value = float(data.get("frequency_penalty", 0.0))
                except Exception:
                    pass

            if data.get("presence_penalty") == "default":
                self.pres_pen_defb.value = True
            else:
                self.pres_pen_defb.value = False
                try:
                    self.pres_pen_sl.value = float(data.get("presence_penalty", 0.0))
                except Exception:
                    pass

            self._render_transcript()
            with self.status_out:
                print(f"Loaded: {short_path} (model={model})")
        except Exception as e:
            with self.log_out:
                print(f"Error loading chat: {e}")
