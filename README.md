# Dramography
### Who Knew What When

Welcome to **Dramography**, the narrative engine that tears the cover off your favorite novels to show you the gears turning underneath. 

Ever wanted an "exploded view" diagram of a plot? Ever argued about who knew what and when? Dramography uses LLMs to generate **cutaway views** of your story, mapping the invisible mechanics of character intentions, lies, and the physical movement of the MacGuffin.

---

## What does it do?

Dramography takes a raw text file (a novel) and runs it through a 5-step pipeline to extract the "truth" of the story. It separates what characters *believe* from what actually *happened*.

It generates two main artifacts:

### 1. The God View 
This is the omniscient truth. It tracks:
*   **The Action:** What actually happened.
*   **The Gap:** Who is wrong? (e.g., Alice thinks Bob is flirting; Bob is actually trying to steal her wallet).
*   **The Stakes:** What happens if the truth comes out?

### 2. The Swim Lanes 
This is the logistics tracker. It tracks:
*   **Timeline:** Minute-by-minute breakdowns.
*   **Locations:** Where is everyone standing?
*   **The Objects:** Who is holding the "Cursed Ruby" right now?

---

## The Pipeline (How the sausage is made)

We don't just ask the AI once. We grill it. Repeatedly.

1.  **Summarize:** We chew the book into bite-sized chapter summaries.
2.  **Consolidate:** We merge characters (so "Dave" and "David" act as one) and build a master grammar.
3.  **Detail Extraction:** We go back in for the juicy bits.
4.  **The Matrix (God & Swim):** We generate the deep data JSONs for every chapter.
5.  **The Mega-Merge:** We stitch it all together into one massive JSON file representing the entire narrative arc.

---

## How to Run It

### Prerequisites
*   Python 3.x
*   A local LLM server (like `llama.cpp`) running and ready to rumble.
*   A book in `.txt` format (UTF-8, please!).

### Step 1: Configure
Edit your `config.yaml`. Tell us where your book is and where your LLM lives.

```yaml
processing:
  txt_file: "path/to/my_mystery_novel.txt"
  chapter_split_string: "### CHAPTER ###"

steps:
  step4_godview:
    godview_prompt: "..."  # The magic words
    swimlane_prompt: "..." # The other magic words
```

### Step 2: The Magic Button
Run the pipeline. You can run the whole thing, or just specific steps if you're tweaking prompts.

```bash
# Run the whole shebang
python run_pipeline_steps.py config.yaml

# Or, just run the God View generation (if previous steps are done)
python run_pipeline_steps.py config.yaml --step 4
```

### Step 3: View the Gears
Check your `results/` folder. You'll find `god_view_combined.json` and `swim_lane_combined.json`.
Open `narrative_viewer.html` and load in the results to inspect it.

**Example Output (JSON snippet):**
```json
{
  "scene_id": "ch2_sc3",
  "action_truth": "The butler puts the poison in the tea.",
  "information_gaps": [
    {
      "character": " The Duke",
      "belief": "Thinks the tea smells like almonds because it is fancy."
    }
  ],
  "stakes": "If the Duke drinks, the inheritance is lost."
}
```

---

Happy Mapping!