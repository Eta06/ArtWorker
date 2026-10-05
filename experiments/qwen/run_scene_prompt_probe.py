"""One-variable track3 prompt control using the unchanged growing sampler.

Uses the edge-context arm of run_geometry_trial. The additional scene
instructions are a manually authored research prompt, not automatic structural
conditioning. Existing weights, masks, source, noise and six-step schedule are
unchanged. Always pass a fresh --output and --arms edge-context.
"""
import run_geometry_trial as runner
from run_trial import PROMPT

runner.PROMPT = PROMPT + (
    " Structural continuity is essential: the metal roadside guardrail is one"
    " continuous unbroken object. Continue the incoming angle of its metal rails"
    " smoothly across the original image boundary, without an elbow, sideways"
    " displacement, extra joint, or change of rail thickness. The guardrail"
    " reaches the original square's top edge around sixty percent of its width;"
    " its continuation goes upward and rightward and leaves the right edge soon"
    " above the original square. Maintain the same steep overhead camera view."
    " The added upper region consists of nearby water, riverbank vegetation and"
    " road; continue their existing scale and twilight exposure. The lower region"
    " is the same asphalt, with continuous grain and exposure. No sky, horizon,"
    " distant landscape, horizontal color bands, or abrupt texture changes."
)

if __name__ == '__main__':
    runner.main()
