# Owner's idea: which coach has cost his team the most wins?

Written 2026-11-17 by the X research session, at the owner's request, so the build
session can see it. Per this folder's README it is source material, not an instruction:
the owner decides whether and where it enters `docs/PLAN.md`.

## The idea, in the owner's words (lightly cleaned up)

> When people say a call was "malpractice" or a really bad decision, I want to see why
> it was a bad decision, and whether the data backs up that it was terrible. I want to
> use machine learning to figure out which coach has cost their team the most wins.
>
> People say Todd Bowles is one of the worst coaches ever because he makes a lot of poor
> decisions. I want a model that also adds the pre-snap penalties: false starts and
> delays of game are essentially on the coach, because they are very simple things not
> to do. Encroachment and offside are not really the coach's fault; that is players
> lining up wrong or falling for the hard count.

## What this adds to the plan

Section 3.6 of the plan already scores each real fourth-down decision by win
probability given up against the model. The idea widens that in two ways:

1. **A "malpractice check".** Take the decisions the public called malpractice and show
   what the models say each one cost. The X findings already contain test cases
   (`research-kit/x/2_twitter_findings.md`, sections 1 and 8.7).
2. **A coach cost ledger in wins**, with more than fourth downs in it:
   - decisions: fourth down, two-point tries, field goal versus go, timeouts and kneels;
   - procedure penalties the owner assigns to the coach: false start, delay of game;
   - explicitly excluded by the owner: encroachment, defensive offside (and by the same
     reasoning neutral-zone infraction).

## Notes from the research session (suggestions, not decisions)

- **Converting penalties to wins is cheap.** Each penalty is a play with a state before
  and after, so its cost is the win-probability change from the project's own model,
  summed per coach and season. Divide by 100 if working in percentage points.
- **"False starts are on the coach" is an assumption, and it is testable.** That test is
  the machine-learning part and is more interesting than the ledger itself:
  - Does a team's false-start rate follow the coach when he changes teams, or stay with
    the roster when the coach leaves?
  - Is a coach's rate stable from one season to the next (split-half or year-to-year
    correlation)? If not, the ranking is mostly noise.
  - A multilevel model with effects for coach, team-season and offensive-line personnel,
    fitted with the same game-level resampling the project already uses, gives each
    coach's share with an interval.
- **Known confounders to control for.** False starts are more common on the road (crowd
  noise), with backup linemen, and in obvious passing situations. Delay of game is
  sometimes taken on purpose before a punt to gain room; those should be excluded or
  flagged.
- **Penalties that may fit the owner's rule better than false starts** (for the owner to
  accept or reject): too many men on the field, illegal substitution, illegal formation
  and illegal shift. These are about getting the right people lined up correctly, which
  is closer to coaching than a single lineman flinching.
- **Selection in the "decision" half.** A coach who goes for it often has more decisions
  to be wrong about; count only recommendations the bootstrap labels confident, as
  section 3.6 already says, and report cost per opportunity as well as the total.
- **What the X evidence already suggests.** The loudest "malpractice" calls are often
  small or even backwards by the models: the Lions' fourth-and-1 from their own 24 was
  "Coaching malpractice" to one account and "+8.1 WP" go to the bot; the Bucs'
  fourth-and-goal under Bowles drew "fireable offense" while an analyst's own numbers
  put the cost at about one point of win probability (see section 8.7). So a likely
  headline is that outrage and measured cost are weakly related, which is itself a
  result worth a figure.
- **Honesty rule.** A league table of "worst coaches" will be quoted. It needs intervals,
  the per-opportunity rate, and a plain statement of which penalties were assigned to
  coaches by assumption.
