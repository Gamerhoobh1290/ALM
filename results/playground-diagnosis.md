# Playground diagnosis (fixed seed 42, identical settings, CPU, no training)

Settings: tokens=60 temp=0.8 top_k=40 seed=42 chat=true

## Replies
### A-default-1M :: 'hi'
A: "I'm afraid it's not . And I'm afraid I'm afraid it's just old . But I don't think it's too fast ."
checks: {'non_empty': True, 'no_replacement_char': True, 'printable_ratio': 1.0, 'no_prompt_echo': True, 'no_runaway_repeat': True}

### A-default-1M :: 'hello'
A: "I'm afraid it's not . And I'm afraid I'm afraid it's just old . But I don't think it's too fast ."
checks: {'non_empty': True, 'no_replacement_char': True, 'printable_ratio': 1.0, 'no_prompt_echo': True, 'no_runaway_repeat': True}

### A-default-1M :: 'Good morning! How are you today?'
A: "I'm the PATHATICO WATIC DENCULESTING: Oh my name is STLSTIONALTION."
checks: {'non_empty': True, 'no_replacement_char': True, 'printable_ratio': 1.0, 'no_prompt_echo': True, 'no_runaway_repeat': True}

### A-default-1M :: 'What does the Windows Recycle Bin do?'
A: 'I know, how to identify the company’s resources and the use of the company in the area. In case the company, there are five types of companies with European'
checks: {'non_empty': True, 'no_replacement_char': True, 'printable_ratio': 1.0, 'no_prompt_echo': True, 'no_runaway_repeat': True}

### A-default-1M :: 'Name three red fruits.'
A: "Name one is a red fruit.\nIs it a good idea for your meal. It's the perfect food. You can have it with a few sugar. You can also be sure if you have enough to eat it."
checks: {'non_empty': True, 'no_replacement_char': True, 'printable_ratio': 1.0, 'no_prompt_echo': True, 'no_runaway_repeat': True}

### B-newer-8.29M :: 'hi'
A: 'yes'
checks: {'non_empty': True, 'no_replacement_char': True, 'printable_ratio': 1.0, 'no_prompt_echo': True, 'no_runaway_repeat': True}

### B-newer-8.29M :: 'hello'
A: "I'm afraid it's not so bad.Is it safe?\nIt's not very warm and warm.But I'll have a good idea ."
checks: {'non_empty': True, 'no_replacement_char': True, 'printable_ratio': 1.0, 'no_prompt_echo': True, 'no_runaway_repeat': True}

### B-newer-8.29M :: 'Good morning! How are you today?'
A: "I'm really glad to meet you too.I'm glad to hear that.It 'hi, you mean to go to the park and look out for my name.And what does it mean to be in there too much.So you know, I think I can go to the beach and"
checks: {'non_empty': True, 'no_replacement_char': True, 'printable_ratio': 1.0, 'no_prompt_echo': True, 'no_runaway_repeat': True}

### B-newer-8.29M :: 'What does the Windows Recycle Bin do?'
A: 'The Windows Van Denied is a U.S. Department of Honour. It is a U.S. School of Chicago and is a U.S. educational institution and is an aut'
checks: {'non_empty': True, 'no_replacement_char': True, 'printable_ratio': 1.0, 'no_prompt_echo': True, 'no_runaway_repeat': True}

### B-newer-8.29M :: 'Name three red fruits.'
A: "I'm really glad to ."
checks: {'non_empty': True, 'no_replacement_char': True, 'printable_ratio': 1.0, 'no_prompt_echo': True, 'no_runaway_repeat': True}

### C-earlier :: 'hi'
A: 'I know .'
checks: {'non_empty': True, 'no_replacement_char': True, 'printable_ratio': 1.0, 'no_prompt_echo': True, 'no_runaway_repeat': True}

### C-earlier :: 'hello'
A: 'I know .'
checks: {'non_empty': True, 'no_replacement_char': True, 'printable_ratio': 1.0, 'no_prompt_echo': True, 'no_runaway_repeat': True}

### C-earlier :: 'Good morning! How are you today?'
A: "I'm really starting to have a good dinner ."
checks: {'non_empty': True, 'no_replacement_char': True, 'printable_ratio': 1.0, 'no_prompt_echo': True, 'no_runaway_repeat': True}

### C-earlier :: 'What does the Windows Recycle Bin do?'
A: 'I know, how to ’ Vacuinely Mill Death, please .'
checks: {'non_empty': True, 'no_replacement_char': True, 'printable_ratio': 1.0, 'no_prompt_echo': True, 'no_runaway_repeat': True}

### C-earlier :: 'Name three red fruits.'
A: 'I know, how to ’ t a red fruit I can be very similar to .'
checks: {'non_empty': True, 'no_replacement_char': True, 'printable_ratio': 1.0, 'no_prompt_echo': True, 'no_runaway_repeat': True}

## Run validation (status/summary, no new training)
- A-default-1M `results/sft-assistant/checkpoints/step_00000245.pt`: state=target_reached step=245 tokens=1003520/1000000 train=3.0427000522613525 val=2.6824484553445136 holdout=2.727139237121612
- B-newer-8.29M `results/auto-assistant-20260917-215605-s1/checkpoints/step_00002024.pt`: state=data_stop: SFT train exhausted step=2024 tokens=8290304/8291299 train=2.5170514583587646 val=2.5241722199992958 holdout=2.6104308386702337
- C-earlier `results/sft-assistant/checkpoints/step_00000100.pt`: state=target_reached step=245 tokens=1003520/1000000 train=3.0427000522613525 val=2.6824484553445136 holdout=2.727139237121612

## Readout (fixed-seed fair comparison, human judgment)
- Does B beat A on greetings? Marginally: C `I know .` → A repeats one afraid/old template → B varies (`yes` / safe/warm / glad-to-meet-you) but stays generic. Not a usable greeting yet.
- Does B beat A on basic instructions? No genuine win: Recycle Bin `company resources` → `Windows Van Denied … Chicago` hallucination; red fruits `Name one is a red fruit…` → `I'm really glad to .` collapse. Loss improved (val 2.68→2.52, holdout 2.72→2.61) without capability win — matches prior `215523` promotion hold (conv F1 regressed).
- Recommendation for one targeted next experiment (no auto-launch): keep stage-2 paused; first run a tiny overfit probe (50 DailyDialog greetings to loss<1.0) to prove the 8.8M SFT stack can memorize the chat template at all. If it cannot, fix data/template/LR before spending the remaining 1.71M budget; if it can, resume stage-2 from `step_00002024.pt` with the same fixed-seed eval gate.
