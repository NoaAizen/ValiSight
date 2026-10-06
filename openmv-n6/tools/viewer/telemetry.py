"""Sensor health and timing summaries for the live viewer's HTTP API."""
import os

import live_channels
import soc as hostsoc
from .constants import HEAP_FLOOR, LEPTON_SAFE_GAP_MS, DETECT_STALE_S

# ---------------------------------------------------------------- health
#
# The offline suites prove the pipeline is correct on frames that sit still. None
# of them can tell you whether the thing in front of you right now is producing
# numbers worth writing down - that depends on the sensor's state, the range it
# auto-picked, whether the link is tearing, and whether a warp was ever loaded.
#
# So these are the checks that can only be made live, and the distinction the
# levels draw is deliberately about *trust in the reading*, not about tidiness:
#
#   fail  the numbers on screen are wrong or absent. Do not record anything.
#   warn  the numbers are usable but qualified, and the qualification changes how
#         they should be read - not a nag to be cleared.
#   ok    nothing known to be wrong.
#
# A green panel is not a claim that the measurement is accurate. It is a claim
# that none of the failures this code can see are happening.

STALE_S = 5.0           # no frame for this long and the stream is considered dead
# The sensor's own rate; the pipeline cannot beat it. Measured on the board
# 2026-08-06 over 5220 frames: the inter-frame interval is a three-valued delta
# function - 113 ms (326x), 114 ms (4742x), 115 ms (148x), and 1824 ms on the
# three FFCs. Median 114 ms = 8.772 fps, not the 8.82 the datasheet implies.
#
# The 70% threshold below is weaker than it looks, and the constant cannot fix
# that on its own: whether it can fire at all depends on the fps averaging
# window. Over that same run a 10-frame window dipped to 3.5 fps, a 30-frame
# window to 5.8, and a 100-frame window never below 7.6. So with a long window
# it never fires; with a short one it fires only on FFC gaps, which makes it an
# accidental FFC detector wearing a link-health label. Nothing else in ten
# minutes came within 25% of it.
LEPTON_FPS = 8.772


def health(pipe, state, now):
    """Returns a list of {name, level, text}. Pure enough to test off a dict."""
    out = []

    def add(name, level, text):
        out.append({"name": name, "level": level, "text": text})

    # --- the link
    err = state.get("error")
    last = state.get("last_frame_t")
    fps = state.get("fps", 0.0)
    restarting = state.get("restarting_since")
    link_recovering = state.get("link_recovering_since")
    if restarting is not None:
        # Recovering is not failing. A restart takes ~13s against a 5s stale
        # threshold, so without this branch every successful defence reports as a
        # dead stream - which is how a health panel gets ignored.
        add("stream", "warn", "restarting the board: %s (%.0fs)"
            % (state.get("last_restart", "fault"), now - restarting))
    elif link_recovering is not None:
        add("stream", "warn", "dropping one incomplete batch: %s"
            % state.get("link_recovering_reason", "USB short write"))
    elif err:
        add("stream", "fail", err)
    elif last is None:
        add("stream", "warn", "waiting for the first frame")
    elif now - last > STALE_S:
        add("stream", "fail", "no frame for %.0f s" % (now - last))
    elif fps < LEPTON_FPS * 0.7:
        add("stream", "warn", "%.1f fps, expected %.1f" % (fps, LEPTON_FPS))
    else:
        add("stream", "ok", "%.1f fps" % fps)

    # --- framing. A resync means bytes went missing on the wire and a frame was
    # dropped to get back in step. The stream survives it, which is the point,
    # but silently surviving is how this went unnoticed for so long - a link
    # that needs to resynchronise is not a healthy link, so say so.
    rs = state.get("resyncs", 0)
    last_rs = state.get("last_resync_t")
    if rs and last_rs is not None and now - last_rs < 60.0:
        add("framing", "warn", "%d resync%s: %s" % (
            rs, "" if rs == 1 else "s", state.get("last_resync", "")))
    else:
        add("framing", "ok", "in step%s" % (
            " (%d recovered total)" % rs if rs else ""))

    # --- the two sensors' timing, from the board's clock. Separate from the
    # "stream" check above on purpose: that one is about the link keeping up,
    # this one is about whether the pair being fused was looked at at the same
    # moment. A link can be perfect while the pairing is not.
    tm = timing(state, now)
    if tm["thermal_ms"]:
        med = tm["thermal_ms"]["median"]
        want = tm["expected_ms"]
        if abs(med - want) > 0.25 * want:
            add("cadence", "warn", "thermal frame every %d ms, expected %.0f" % (med, want))
        else:
            add("cadence", "ok", "thermal %d ms (%.2f fps)" % (med, tm["thermal_fps"]))
    if tm["skew_frac"] is not None:
        sm = tm["skew_ms"]["median"]
        if tm["skew_frac"] > SKEW_WARN:
            add("pairing", "warn", "visible frame lags thermal by %d ms (%.0f%% of a "
                "frame) - anything moving is displaced before the warp sees it"
                % (sm, 100 * tm["skew_frac"]))
        else:
            add("pairing", "ok", "visible +%d ms after thermal (%.0f%% of a frame)"
                % (sm, 100 * tm["skew_frac"]))
    # --- the sensor being starved by host load. This is the check the whole
    # board clock earns its keep on. Everything else about a slow host is a
    # frame rate complaint; this one is the part that cannot be undone, because
    # a wedged Lepton needs a fresh csi.CSI() object and no amount of retrying,
    # re-arming or soft re-init has ever recovered one.
    if tm["starved"]:
        add("thermal load", "fail",
            "%d thermal gap%s past %dms, worst %dms - the board was not being "
            "drained and the sensor went unserviced beyond anything measured safe"
            % (tm["starved"], "" if tm["starved"] == 1 else "s",
               tm["safe_gap_ms"], tm["last_starve_ms"] or 0))
    elif tm["stalls"]:
        add("thermal load", "warn",
            "%d write stall%s - the host stopped draining the port for 500ms at a "
            "time. Frames are lost and the sensor waits; reduce host work before "
            "this reaches %dms" % (tm["stalls"], "" if tm["stalls"] == 1 else "s",
                                   tm["safe_gap_ms"]))
    elif tm["thermal_ms"]:
        # The longest gap excluding FFCs. An FFC is a different mechanism and
        # not a hazard: the sensor stops delivering for 1824ms but snapshot()
        # BLOCKS, so the consumer stays parked in the driver rather than leaving
        # the part unserviced. Quoting it here would read as the sensor being
        # three quarters of the way to danger every three minutes.
        add("thermal load", "ok", "no write stalls, longest gap %dms of %dms"
            % (tm["thermal_ms"]["max"], tm["safe_gap_ms"]))

    # --- the machine this is running on. Deliberately next to the check above:
    # they are the same subject seen from opposite ends. "thermal load" is the
    # damage, measured on the board's own clock and therefore already in the
    # past by the time it appears; these are the causes, and they move first.
    s = state.get("soc")
    if s is not None:
        out.extend(hostsoc.checks(s.read()))

    if tm["ffc_ago_s"] is not None and tm["ffc_ago_s"] < 3.0:
        # The sensor needs time to settle after the shutter. Quoting through one
        # is the kind of error that survives into a report.
        add("ffc", "warn", "shutter closed %.1fs ago - let the sensor settle "
            "before quoting a reading" % tm["ffc_ago_s"])

    # --- short or unreadable JPEGs. The visible signature of bytes lost on the
    # wire: the board announced a length and the host got fewer usable bytes.
    # Framing can stay in step through this, so it does not always show up as a
    # resync - and a frame that silently vanishes makes the link look healthier
    # than it is. The known cause on this bench is another process opening the
    # port and toggling DTR, which makes the CDC discard what it has queued;
    # keep ModemManager off the device (ID_MM_DEVICE_IGNORE) before blaming the
    # firmware.
    bad = state.get("bad_jpeg", 0)
    if bad:
        add("jpeg", "warn", "%d frame%s arrived corrupt - bytes lost on the wire"
            % (bad, "" if bad == 1 else "s"))

    # --- the render thread. Dropping is by design when it falls behind, but the
    # viewer must say so: a page showing every third frame while quoting the
    # board's fps is describing a stream nobody is watching.
    dropped, rendered = state.get("dropped", 0), state.get("rendered", 0)
    if rendered:
        rate = dropped / float(dropped + rendered)
        if rate > 0.25:
            add("render", "fail", "%.0f%% of frames dropped before display" % (100 * rate))
        elif rate > 0.02:
            add("render", "warn", "%.0f%% dropped before display" % (100 * rate))
        else:
            add("render", "ok", "%d rendered" % rendered)

    # --- detection. Reported separately from the boxes themselves because the
    # thing worth knowing is whether the temperature beside a label means
    # anything, and without a calibrated warp it does not.
    # --- scene light. Its own line, above the detector, because it is what
    # decides how to read the detector's line: a detector reporting nothing is
    # either a working detector in an empty scene or a blind one, and those are
    # the same sentence until this number is beside it.
    light = state.get("light")
    if light is not None:
        if light["state"] == "blind":
            add("light", "warn",
                "visible is BLIND - %.1f%% of the frame is lit. Measured at "
                "this light the detector recovers 1.6%% of the people in "
                "front of it; thermal and radar are the channels to read"
                % (100 * light["lit"]))
        elif light["state"] == "dim":
            add("light", "warn",
                "part-lit - %.0f%% of the frame is lit, and nobody has measured "
                "what this detector finds there. Its silence is NOT excused"
                % (100 * light["lit"]))
        else:
            add("light", "ok", "lit - %.0f%% of the frame, median luma %.0f"
                % (100 * light["lit"], light["p50"]))

    dt = state.get("detect_t")
    if dt is not None:
        age = now - dt
        n_det = len(state.get("detections") or [])
        blind = (light or {}).get("state") == "blind"
        if state.get("detect_error"):
            add("detect", "fail", state["detect_error"])
        elif age > DETECT_STALE_S * 4:
            add("detect", "warn", "no detection for %.0fs" % age)
        elif blind and not n_det:
            # The one case where finding nothing is not a result. Saying "0
            # boxes" here reads as "nobody is there", which is the claim this
            # rig has no light to make.
            add("detect", "warn", "silent, and explained: no light to see by")
        elif state.get("detections_rejected"):
            rej = state["detections_rejected"]
            add("detect", "warn", "%d usable box%s; rejected %d implausible "
                "person box%s (%s)" % (
                    n_det, "" if n_det == 1 else "es", len(rej),
                    "" if len(rej) == 1 else "es",
                    rej[0].get("rejected", "geometry")))
        elif not pipe.warped:
            add("detect", "warn", "%d box%s, %.0fms - temperatures are read through "
                "the placeholder warp and belong to the wrong pixels"
                % (n_det, "" if n_det == 1 else "es", state.get("detect_ms", 0)))
        else:
            add("detect", "ok", "%d box%s, %.0fms"
                % (n_det, "" if n_det == 1 else "es", state.get("detect_ms", 0)))

    # --- the radar link. Separate from the "N in / M out" the overlay prints
    # on the frame: once nothing arrives the overlay has nothing to draw, and a
    # silently absent sensor looks exactly like an empty scene.
    if state.get("radar_proj") is not None:
        if state.get("radar_error"):
            add("radar", "fail", state["radar_error"])
        elif not state.get("radar_frames"):
            add("radar", "warn", "no radar frame yet - was the config sent? "
                "(tools/send_radar_cfg.py)")
        else:
            drop = state.get("radar_dropped", 0)
            add("radar", "warn" if drop else "ok",
                "%d frames, %d drawn / %d offscreen%s"
                % (state["radar_frames"], state.get("radar_drawn", 0),
                   state.get("radar_offscreen", 0),
                   ", %d bytes dropped" % drop if drop else ""))

    # --- selected product channel. An unavailable sensor must not look like an
    # empty scene, and the current visible transport must not be advertised as
    # colour merely because the stable API id is rgb_radar.
    product = live_channels.get(pipe.channel)
    if product.raw_radar and state.get("radar_proj") is None:
        add("channel", "warn", "%s selected without a radar source"
            % product.label)
    elif pipe.channel == "rgb_radar":
        add("channel", "warn", "visible + radar is PAG7936 luma in this build; "
            "true RGB waits on the RGB565 bandwidth gate")
    # An AI overlay is optional decoration on the thermal-bearing products;
    # missing inference must not make a healthy thermal camera look broken.
    # The dedicated AI product, however, has no useful contract without an
    # inference source, so it advertises that absence explicitly.
    elif pipe.channel == "ai" and not (state.get("detector_available")
                                      or state.get("students") is not None):
        add("channel", "warn", "AI selected but neither detector nor student "
            "engines are available")
    else:
        add("channel", "ok", "%s" % product.label)

    # --- the AI channels. Not "is the model loaded" - the switches are live,
    # so the question worth answering is whether what is on the screen right now
    # is what the operator thinks they are looking at. A fusion channel that
    # cannot pair draws nothing, and nothing looks exactly like an empty scene.
    st = state.get("students")
    if st is not None:
        n_t = len(state.get("student_thermal") or [])
        n_r = len(state.get("student_radar") or [])
        n_f = len(state.get("student_fused") or [])
        on = [n for n, flag in (("thermal", pipe.ai_thermal),
                                ("radar", pipe.ai_radar),
                                ("fusion", pipe.ai_fusion)) if flag]
        no_lut = st.get("th2vis") is None
        no_radar = st.get("radar") is None
        if state.get("student_error"):
            add("ai", "fail", state["student_error"])
        elif not on:
            add("ai", "warn", "all three channels switched off - the students "
                "are loaded and nothing is running")
        elif pipe.ai_fusion and (no_lut or no_radar):
            add("ai", "warn", "fusion is on but cannot pair: %s"
                % ("no warp LUT, so thermal boxes never reach the visible plane"
                   if no_lut else "no radar student engine on this run"))
        elif pipe.ai_thermal and no_lut:
            add("ai", "warn", "thermal channel has no warp LUT - %d box%s found, "
                "none can be placed on the picture"
                % (n_t, "" if n_t == 1 else "es"))
        else:
            seen = state.get("student_seen") or {}
            rejected_f = state.get("student_fused_rejected") or []
            hidden = (max(0, seen.get("thermal", 0) - n_t)
                      + max(0, seen.get("radar", 0) - n_r))
            add("ai", "warn" if rejected_f and pipe.ai_fusion else "ok",
                "%s on - T %d / R %d / TR %d%s%s"
                % ("+".join(on), n_t, n_r, n_f,
                   "" if not hidden else
                   ", %d below the confidence floor or deduplicated" % hidden,
                   "" if not rejected_f else
                   ", %d TR rejected by raw-radar/geometry gate" % len(rejected_f)))

    # --- the lock. On its own line rather than inside the detector's: the
    # question it answers is not "did a sensor see somebody" but "is the viewer
    # still holding the person it had", and a lock held only by a coast is a
    # different claim from a lock being measured.
    if pipe.ai_lock:
        tr = state.get("tracks") or []
        static = state.get("tracks_static") or []
        coasting = [t for t in tr if t.get("coasting")]
        held = ", %d warm and motionless (not drawn)" % len(static) if static else ""
        if not tr:
            add("lock", "ok", "nothing locked" + held)
        elif len(coasting) == len(tr):
            add("lock", "warn", "%d track%s, all coasting - no sensor has "
                "confirmed them this cycle (worst %.1fs)"
                % (len(tr), "" if len(tr) == 1 else "s",
                   max(t["coasting"] for t in coasting)))
        else:
            add("lock", "ok", "%d locked%s%s"
                % (len(tr) - len(coasting),
                   ", %d coasting" % len(coasting) if coasting else "", held))

    # --- recording. A recorder that died mid-session must not be discovered at
    # the end of the campaign; the mp4 writer's failure mode is a 0-byte file.
    rec = state.get("recording")
    if rec:
        if state.get("video_error"):
            add("recording", "fail", state["video_error"])
        else:
            add("recording", "ok", "%d frames -> %s"
                % (state.get("video_frames", 0), os.path.basename(rec)))

    # --- the map layer (Yael's mapinit, via perception.map_api). Only present
    #     when --map was given. A failed stage is reported with its name because
    #     "map failed" has two unrelated fixes: install the EGM2008 grid, or
    #     fetch the priors for this operating area.
    m = state.get("map")
    if m is not None:
        if m.get("pending"):
            add("map", "warn", "initialising at %.4f,%.4f" % (
                m["location"]["latitude_deg"], m["location"]["longitude_deg"]))
        elif m.get("ok"):
            alt = m.get("ego_altitude_prior") or {}
            add("map", "ok", "geoid %.2f m, ground %.0f m, %s" % (
                (m.get("geoid") or {}).get("undulation_m", float("nan")),
                alt.get("orthometric_m", float("nan")),
                "+".join(s["name"] for s in m.get("stages", [])
                         if s["status"] == "ok") or "no stage"))
        elif m.get("error"):
            add("map", "fail", ("deployment: " if m.get("deployment") else "")
                + m["error"].splitlines()[0])
        else:
            failed = [s["name"] for s in m.get("stages", []) if s["status"] == "failed"]
            add("map", "fail", "stage%s failed: %s" % (
                "" if len(failed) == 1 else "s", ", ".join(failed) or "unknown"))

    # --- VoSPI tearing. Unlike the dead rows this really is random, and a torn
    #     frame is stale data in part of the image, not a marked defect.
    #
    #     Measured 2026-08-06: 0 torn frames in 5220 over ten minutes, so on this
    #     link tearing is rare rather than routine - read a green pill here as
    #     the expected state, not as a reassurance. One structural caveat: the
    #     detector evaluates the segment seams at rows 29/59/89, and the dead-row
    #     run 55..63 straddles the 59/60 seam. That seam is therefore measured
    #     across two stuck rows and cannot report a tear there at all.
    win = state.get("torn_window") or []
    if win:
        rate = sum(win) / float(len(win))
        if rate > 0.20:
            add("tearing", "fail", "%.0f%% of frames torn" % (100 * rate))
        elif rate > 0.02:
            add("tearing", "warn", "%.0f%% of frames torn" % (100 * rate))
        else:
            add("tearing", "ok", "%.0f%% torn" % (100 * rate))

    if not pipe.f.have_frame:
        return out

    # --- dead rows. On this unit the count is 0 or 14 and holds for a whole
    #     session, so a count that moves frame to frame is the detector following
    #     the scene instead of the defect - which is the failure the flat+lifted
    #     test was written to avoid, and worth catching live rather than in a
    #     capture review three weeks later.
    hist = state.get("rows_window") or []
    rows = pipe.f.rows_rebuilt
    total = pipe.f.cfg.th_h
    if len(set(hist)) > 1:
        add("dead rows", "fail", "count varies (%s) - the detector is tracking the scene"
            % "/".join(str(v) for v in sorted(set(hist))))
    elif rows == 0:
        add("dead rows", "ok", "none")
    else:
        add("dead rows", "warn", "%d of %d rows rebuilt - readings there are "
            "interpolated" % (rows, total))

    # --- board heap. The one check that predicts a failure instead of reporting
    # one. gc.collect() with the Lepton up wedges the part permanently, so the
    # automatic collector firing on an exhausted heap is fatal - and the heap
    # drains steadily because the streaming loop cannot be made to allocate
    # nothing. The supervisor restarts the bring-up before that happens; this
    # says how much room is left and whether it has had to.
    free = state.get("heap_free")
    restarts = state.get("restarts", 0)
    if free is None:
        add("board heap", "ok", "no reading yet" if restarts == 0 else
            "%d restart%s so far" % (restarts, "" if restarts == 1 else "s"))
    else:
        mb = free / (1 << 20)
        note = "%.1fMB free" % mb
        if restarts:
            note += ", %d restart%s (%s)" % (restarts, "" if restarts == 1 else "s",
                                             state.get("last_restart", "fault"))
        # The floor is where the supervisor acts, so approaching it is normal
        # operation and not worth a warning until it is close enough to be soon.
        add("board heap", "warn" if free < HEAP_FLOOR * 2 else "ok", note)

    # --- registration
    if not pipe.warped:
        add("registration", "warn", "placeholder warp - the thermal layer is "
                                    "stretched, not registered to the visible one")
    else:
        add("registration", "ok", "calibrated warp")

    cov = pipe.cover_grid().mean()
    if cov <= 0.0:
        add("coverage", "fail", "no thermal coverage anywhere")
    elif cov < 0.30:
        add("coverage", "warn", "%.0f%% of the frame has thermal data" % (100 * cov))
    else:
        add("coverage", "ok", "%.0f%% covered" % (100 * cov))

    # --- the sensor range, which is what every reading is scaled by
    lo, hi = state.get("range", (0, 0))
    if hi <= lo:
        add("range", "fail", "no sensor range reported - readings are meaningless")
    else:
        per_code = (hi - lo) / 255.0
        # Measured on this part 2026-08-06, 250-frame runs: NETD is 33 mK, not
        # the ~50 mK the datasheet implies, and it is the same in both gain
        # modes. So a code finer than ~0.033 C resolves noise; much coarser than
        # ~0.2 C and the auto-range has given away resolution it did not need to.
        #
        # NETD is the wrong figure for trusting an *absolute* reading, though.
        # Over the same runs the common-mode-removed temporal spread was
        # 126-148 mK - 4x worse. Two readings seconds apart are comparable to
        # ~0.15 C; NETD only bounds frame-to-frame differencing.
        lvl = "warn" if per_code > 0.2 else "ok"
        add("range", lvl, "%d..%d C, %.3f C/code%s" % (lo, hi, per_code,
            " - re-run auto-range for finer steps" if lvl == "warn" else ""))

    # --- clipping. Counted after the repair so the dead rows are not mistaken
    #     for a saturated scene. Pixels pinned at either end carry no temperature
    #     at all: they are 'at least this hot', which is not a measurement.
    t = pipe.prep_frame()
    clipped = float(((t == 0) | (t == 255)).mean())
    if clipped > 0.05:
        add("clipping", "fail", "%.1f%% of the thermal frame is pinned at the range "
            "ends - those pixels have no temperature" % (100 * clipped))
    elif clipped > 0.005:
        add("clipping", "warn", "%.1f%% pinned at the range ends" % (100 * clipped))
    else:
        add("clipping", "ok", "%.2f%% pinned" % (100 * clipped))

    # --- AGC. Does not touch the readings, which is exactly why it needs saying:
    #     the picture stops being an absolute temperature map while the hover
    #     numbers carry on being correct, and that mismatch is easy to misread.
    if pipe.f.cfg.agc_permille > 0:
        add("agc", "warn", "scene AGC on - tone is scene-relative, readings are not")

    if pipe.eps < 1.0:
        add("emissivity", "warn", "eps %.2f, reflected %.0f C - readings are corrected"
            % (pipe.eps, pipe.refl))

    return out


# ---------------------------------------------------------------- timing
#
# What the two sensors are actually doing, on the board's clock rather than on
# arrival times. The distinction is the whole point: between a frame being
# grabbed and this host seeing it lie a 4KB-chunked CDC write, a 500ms stall
# retry and the host's scheduler, so arrival-time fps measures the link. It has
# been read as a sensor rate more than once in this project.
#
# Three numbers come out of it:
#
#   thermal   the Lepton's cadence. Should be 114ms; it is a three-valued delta
#             function (113/114/115ms) with 1824ms across an FFC.
#   skew      thermal frame in hand -> visible frame in hand. This is the pairing
#             error fusion inherits: the two planes it registers were not looked
#             at at the same instant, so anything moving is displaced by roughly
#             (object speed x skew) before the warp ever sees it.
#   ffc       when the shutter last closed. Readings either side of one are not
#             comparable, which is exactly the kind of thing that is invisible
#             three weeks later in a capture review.


def timing(state, now):
    """Board-side cadence and pairing skew. Pure, so it tests off a dict."""
    dw = [d for d in (state.get("dt_window") or []) if d <= 1000]   # FFC gaps out
    sw = state.get("skew_window") or []
    ffc = state.get("last_ffc_t")

    def stats(v):
        if not v:
            return None
        s = sorted(v)
        return {"median": s[len(s) // 2], "min": s[0], "max": s[-1], "n": len(s)}

    th, sk = stats(dw), stats(sw)
    out = {
        "thermal_ms": th,
        "thermal_fps": round(1000.0 / th["median"], 2) if th and th["median"] else None,
        "skew_ms": sk,
        # The share of one thermal period that the visible frame lags by. This is
        # the number to judge the pairing on - a skew of 12ms means nothing until
        # you know the period is 114ms.
        "skew_frac": round(sk["median"] / float(th["median"]), 3)
                     if th and sk and th["median"] else None,
        "host_fps": round(state.get("fps", 0.0), 2),
        "ffc_ago_s": round(now - ffc, 1) if ffc else None,
        "ffcs": state.get("ffcs", 0),
        "expected_ms": round(1000.0 / LEPTON_FPS, 1),
        # Host load reaching the sensor: 500ms write timeouts the board sat
        # through, and thermal gaps past the envelope the part is measured safe
        # in. Both are about protecting the Lepton, not about frame rate.
        "stalls": state.get("stalls", 0),
        "starved": state.get("starved", 0),
        "last_starve_ms": state.get("last_starve_ms"),
        "safe_gap_ms": LEPTON_SAFE_GAP_MS,
    }
    return out


# Past this share of a thermal period between the two grabs, the pair stops being
# simultaneous in any useful sense. 0.25 of 114ms is 28ms, which at a walking
# 1.4 m/s is 4cm of subject travel - already several thermal pixels at close
# range, and drawn as a registration error rather than as the timing error it is.
SKEW_WARN = 0.25


