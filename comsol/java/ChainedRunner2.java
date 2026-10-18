import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;
import java.nio.file.*;
import java.util.Arrays;
import java.util.List;

/**
 * Chained trajectories with a VARYING per-wet-period environment.
 *
 * WHY (the decisive experiment). The existing longitudinal data cannot answer
 * "does the past carry information the present does not", for two reasons that
 * both defeat it: 90 % of samples sit in the saturated regime where the future
 * increment is zero regardless of history, and both laboratory series run at a
 * CONSTANT environment, so past = present = future forcing and history collapses
 * to "time + regime" by construction.
 *
 * The controlled version needs different HISTORIES under the SAME future forcing.
 * This runner supplies that: the chloride load is a per-period sequence, so two
 * chains can share their final wet period (identical future forcing) while having
 * arrived there from very different earlier periods.
 *
 * Chloride is made time-dependent the same way the wetness envelope is — a sum of
 * smooth windows — so the kinetics keeps its single expression:
 *
 *   Cl(t)    = sum_i Cl_i * W_i(t)
 *   cNa_t(t) = Cl(t)*1[mg/m^2]/(film_h*M_NaCl)   -- kinetics only
 *   cNa_0    = Cl_first/(film_h*M_NaCl)          -- pinned constant: the template's
 *              init/bulk features reference cNa_0 and cannot take a t-dependent value
 *   i0 *= (1 + cNa_t/cCl_ref)^n_cl * Q10^((T(t)-20)/10) * wet(t)
 *
 * Jobs file (SEMICOLON-separated, '#' comments):
 *   outprefix;T_mean_C;film_um;T_amp_C;n_wet;wet_days;cycle_days;tau_sw;cl_list;tlist_days
 * where cl_list is a comma-separated per-period chloride load [mg/m^2], and
 * wet_days is a single value or a comma-separated per-period list; an optional
 * 11th field sets the initial-state chloride load (default: first period's).
 *
 * Usage: ChainedRunner2 <template.mph> <jobs.csv>
 */
public class ChainedRunner2 {
    static final long START_MS = System.currentTimeMillis();
    public static void main(String[] args) {
        try {
            ModelUtil.initStandalone(false);
            Model m = ModelUtil.load("Model", args[0]);
            System.out.println("LOADED " + args[0]);
            int[] ents = m.physics("tcd").feature("es1").selection().entities();
            System.out.println("ELECTRODE_EDGES=" + Arrays.toString(ents));

            m.study().create("stdC");
            m.study("stdC").create("time", "Transient");
            System.out.println("STUDY stdC created (transient only)");

            String[][] chans = {
                {"phil", "potential"}, {"cCl", "chloride"}, {"cFe", "fe2plus"}, {"tcd.pH", "ph"},
            };

            List<String> lines = Files.readAllLines(Paths.get(args[1]));
            for (String ln : lines) {
                if (ln.trim().isEmpty() || ln.startsWith("#")) continue;
                String[] p = ln.trim().split(";");
                String out = p[0];
                try {
                    m.param().set("T_mean", p[1] + "[degC]");
                    m.param().set("film_h", p[2] + "[um]");
                    m.param().set("T_amp", p[3] + "[K]");
                    int nWet = Integer.parseInt(p[4]);
                    // wet_days may be a single value or a comma list (per period)
                    String[] wetStr = p[5].split(",");
                    double cycD = Double.parseDouble(p[6]);
                    double tau = Double.parseDouble(p[7]);
                    String[] clStr = p[8].split(",");

                    // ---- wetness envelope, and a per-period chloride envelope ----
                    StringBuilder w = new StringBuilder(), cl = new StringBuilder();
                    for (int i = 0; i < nWet; i++) {
                        double wetD = Double.parseDouble(wetStr[Math.min(i, wetStr.length - 1)]);
                        double a = i * cycD, b = a + wetD;
                        String win = "0.5*(1+tanh((t-" + a + ")/" + tau + "))"
                                   + "*0.5*(1-tanh((t-" + b + ")/" + tau + "))";
                        if (i > 0) { w.append("+"); cl.append("+"); }
                        w.append(win);
                        double c = Double.parseDouble(clStr[Math.min(i, clStr.length - 1)]);
                        cl.append(c).append("*(").append(win).append(")");
                    }
                    m.param().set("wet_t", w.toString());
                    m.param().set("Cl_t", cl.toString());

                    m.param().set("T", "T_mean+T_amp*cos(2*pi*t)");
                    // cNa_0 is referenced by the template's initial-condition and
                    // bulk-concentration features (init1/conc1/cCl_0) -> it MUST stay
                    // constant. Pin it to the first-period load (same as v1), and give
                    // the kinetics their own time-dependent chloride cNa_t.
                    // optional p[10]: chloride load used for the initial state (default: first period).
                    // Set it equal across chains to remove the cCl-field init legacy as a confound.
                    String cl0Init = (p.length > 10 && !p[10].trim().isEmpty()) ? p[10].trim() : clStr[0];
                    m.param().set("Cl0", cl0Init + "[mg/m^2]");
                    m.param().set("cNa_t", "Cl_t*1[mg/m^2]/(film_h*M_NaCl)");
                    m.param().set("Q10", "2.0");
                    m.param().set("T_ref_Q10", "20[degC]");
                    m.param().set("cCl_ref_cl", "800[mol/m^3]");
                    m.param().set("n_cl", "0.5");
                    String expr = "i0_ref_Fe*(1+eps_i0_Fe)*(tcd.cH/cH_ref)"
                                + "*(1+cNa_t/cCl_ref_cl)^n_cl"
                                + "*Q10^((T-T_ref_Q10)/(10[K]))"
                                + "*wet_t";
                    m.physics("tcd").feature("es1").feature("er1").set("i0_ref", expr);

                    System.out.println("  [stage] params set; cNa_0=" + m.param().get("cNa_0")
                        + " | cNa_t=" + m.param().get("cNa_t"));
                    m.study("stdC").feature("time").set("tlist", p[9]);
                    System.out.println("  [stage] tlist set: " + p[9]);
                    long ts = System.currentTimeMillis();
                    m.geom("geom1").run();
                    System.out.println("  [stage] geom rebuilt (" + (System.currentTimeMillis()-ts)/1000 + "s)");
                    ts = System.currentTimeMillis();
                    m.mesh("mesh1").run();
                    System.out.println("  [stage] mesh rebuilt (" + (System.currentTimeMillis()-ts)/1000 + "s)");
                    System.out.println("  [stage] solve begin @ " + new java.util.Date());
                    System.out.flush();
                    long t0 = System.currentTimeMillis();
                    m.study("stdC").run();
                    long solveMs = System.currentTimeMillis() - t0;
                    System.out.println("  [stage] solve end (" + solveMs/1000 + "s)");

                    String[] tags = m.sol().tags();
                    String useSol = tags[tags.length - 1];
                    try { m.result().dataset("dsReal").set("solution", useSol); }
                    catch (Exception e) {
                        m.result().dataset().create("dsReal", "Solution");
                        m.result().dataset("dsReal").set("solution", useSol);
                        m.result().dataset("dsReal").set("comp", "comp1");
                    }
                    try { m.result().dataset("dsEdge").selection().set(ents); }
                    catch (Exception e) {
                        m.result().dataset().create("dsEdge", "Edge2D");
                        m.result().dataset("dsEdge").selection().set(ents);
                    }
                    try { m.result().dataset("dsEdge").set("data", "dsReal"); } catch (Exception e) {}

                    for (String[] ch : chans) {
                        try {
                            m.result().export().create("eX", "Data");
                            m.result().export("eX").set("data", "dsReal");
                            m.result().export("eX").set("expr", new String[]{ch[0]});
                            m.result().export("eX").set("descr", new String[]{ch[1]});
                            m.result().export("eX").set("filename", out + "_" + ch[1] + ".txt");
                            m.result().export("eX").set("header", "on");
                            m.result().export("eX").run();
                        } catch (Exception e) {} finally {
                            try { m.result().export().remove("eX"); } catch (Exception ig) {}
                        }
                    }
                    try {
                        m.result().export().create("eB", "Data");
                        m.result().export("eB").set("data", "dsEdge");
                        m.result().export("eB").set("expr", new String[]{"tcd.es1.er1.iloc"});
                        m.result().export("eB").set("descr", new String[]{"iloc"});
                        m.result().export("eB").set("filename", out + "_iloc.txt");
                        m.result().export("eB").set("header", "on");
                        m.result().export("eB").run();
                        m.result().export().remove("eB");
                    } catch (Exception e) {}

                    System.out.println("JOB_OK," + out + "," + (solveMs / 1000));
                } catch (Throwable e) {
                    StringBuilder sb = new StringBuilder("JOB_FAIL," + out + ",");
                    sb.append(e.getClass().getName()).append("; ");
                    String sm = e.getMessage();
                    if (sm != null) sb.append(sm.replace('\n', ' ').replace(',', ';'));
                    for (String mName : new String[]{"getTranslatableMessageArray", "getMessageArray"}) {
                        try {
                            Object arr = e.getClass().getMethod(mName).invoke(e);
                            if (arr instanceof String[]) {
                                for (String msg : (String[]) arr) {
                                    sb.append(" ;; ").append(msg.replace('\n', ' ').replace(',', ';'));
                                }
                            }
                        } catch (Throwable ig) {}
                    }
                    System.out.println(sb.toString());
                    System.out.println("  [stage] elapsed-since-start-ms: "
                        + (System.currentTimeMillis() - START_MS));
                    e.printStackTrace(System.err);
                }
                System.out.flush();
            }
            System.out.println("RUNNER_DONE");
            System.out.flush();
            System.exit(0);
        } catch (Exception e) {
            String s = e.getMessage();
            System.out.println("FATAL: " + (s == null ? e.getClass().getName()
                    : s.replace('\n', ' ').substring(0, Math.min(400, s.length()))));
            System.exit(1);
        }
    }
}
