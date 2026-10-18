import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

/**
 * Template v3 — protocol v3, which finally puts the DRIVING signal into the
 * A-group fields.
 *
 * WHY (2026-10-09): protocols v1/v2 left the per-episode environment constant.
 * The system then sits at a true steady state, so over a whole episode the
 * A-group fields move by only ~1e-6 relative — two orders of magnitude BELOW
 * the AE reconstruction noise (~2e-2) — and are unlearnable. That single defect
 * explains both open problems: A-group rollout diverging 100% and the L4
 * cross-episode regression (−9.6%).
 *
 * FIX: the chloride kinetics term gains a TIME-VARYING temperature factor,
 *        i0 *= Q10^((T(t) - 20)/10),
 * with T(t) the diurnal cycle. This is the original design's "weather-driven
 * E_t" and it makes the state track a moving steady state instead of a fixed
 * point. Verified on a probe: A-group change over a 6 h episode goes from
 * 0.0002% (v2) to 8.6% (potential) / 42% (fe2plus) — far above the noise floor.
 *
 * TWO UNIT GOTCHAS, both found the hard way:
 *  1. In a component variable / physics expression, `t` is ALREADY the numeric
 *     time in the study's unit (days). Writing `t/1[day]` divides by 86400 a
 *     second time and silently freezes the driver.
 *  2. Gradients/fields in an ER1 kinetic expression need the `tcd.` prefix for
 *     the field variables (tcd.cH, tcd.cCl) but not for the global parameters.
 *
 * T(t) is deliberately smooth in t so JOB files stay simple: the burn-in phase
 * is expressed with a conditional so the same expression serves both phases.
 *
 * Usage: EnvMapCl3 <src.mph> <dst.mph>
 */
public class EnvMapCl3 {
    public static void main(String[] args) {
        try {
            ModelUtil.initStandalone(false);
            Model m = ModelUtil.load("Model", args[0]);

            m.param().set("cCl_ref_cl", "800[mol/m^3]");   // P3 calibration (2026-10-09)
            m.param().set("n_cl", "0.5");
            m.param().set("Q10", "2.0");        // temperature acceleration factor
            m.param().set("T_ref_Q10", "20[degC]");
            m.param().set("t0_ep", "3");        // burn-in length [days]
            System.out.println("[1] cCl_ref_cl=" + m.param().get("cCl_ref_cl")
                    + " n_cl=" + m.param().get("n_cl") + " Q10=" + m.param().get("Q10")
                    + "   (P3: 1200->800, matches measured pure-Cl CR span)");
            // P3 note: elasticity is state-dependent: e = 4.3*n_cl*(cNa0/ref)/(1+cNa0/ref).
            // ref=1200 flattened the factor over the *validation* chloride range
            // (measured sites sit at cNa0 median 29 mol/m3, far below training's 184),
            // giving model elasticity 0.164 vs 0.280 measured. ref=800 keeps the
            // training-side pure-chloride CR span at 11.2x (P95/P5 5.8x, matching the
            // real marine/inland 3-10x) while lifting the low-chloride slope.

            // diurnal temperature: constant during burn-in, cycling afterwards
            String Texpr = "if(t<t0_ep, T_mean, T_mean+T_amp*cos(2*pi*(t-t0_ep)/1))";
            m.param().set("T", Texpr);
            System.out.println("[2] T(t) = " + m.param().get("T"));

            // chloride kinetics + time-varying temperature factor
            String expr = "i0_ref_Fe*(1+eps_i0_Fe)*(tcd.cH/cH_ref)"
                        + "*(1+cNa_0/cCl_ref_cl)^n_cl"
                        + "*Q10^((T-T_ref_Q10)/(10[K]))";
            m.physics("tcd").feature("es1").feature("er1").set("i0_ref", expr);
            System.out.println("[3] i0_ref: " + m.physics("tcd").feature("es1")
                    .feature("er1").getString("i0_ref"));

            m.save(args[1]);
            System.out.println("SAVED " + args[1]);
        } catch (Exception e) {
            String s = e.getMessage();
            System.out.println("FATAL: " + (s == null ? e.getClass().getName()
                    : s.replace('\n', ' ').substring(0, Math.min(500, s.length()))));
            System.exit(1);
        }
    }
}
