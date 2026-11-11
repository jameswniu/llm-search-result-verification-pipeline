# Keyless demo. Run from the repo root: vhs tools/demo.tape
Output media/demo.gif
Set Shell "bash"
Set FontFamily "Menlo"
Set FontSize 21
Set Width 1200
Set Height 520
Set Padding 24
Set Framerate 12
Set TypingSpeed 35ms
Set Theme { "name": "graphite", "black": "#0e1117", "red": "#b5654a", "green": "#4f9a6f", "yellow": "#8b949e", "blue": "#7f9bb8", "magenta": "#8b949e", "cyan": "#7f9bb8", "white": "#e6e9ee", "brightBlack": "#2c333d", "brightRed": "#b5654a", "brightGreen": "#4f9a6f", "brightYellow": "#8b949e", "brightBlue": "#7f9bb8", "brightMagenta": "#8b949e", "brightCyan": "#7f9bb8", "brightWhite": "#e6e9ee", "background": "#0e1117", "foreground": "#e6e9ee", "selection": "#2c333d", "cursor": "#7f9bb8" }

Hide
Type "export PS1='\[\e[38;2;139;148;158m\]$ \[\e[0m\]' && unset OPENAI_API_KEY && clear"
Enter
Sleep 400ms
Show

Type `python3 run.py sample/catalog.json --cache-dir sample/cache --replay \`
Enter
Type "  2>/dev/null | head -13"
Enter
Sleep 6s

Hide
Type "clear"
Enter
Sleep 400ms
Show
Type "python3 tools/check_replay.py"
Enter
Sleep 2.5s
Type "python3 tools/facts.py --short"
Enter
Sleep 6s
