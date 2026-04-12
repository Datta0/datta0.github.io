"""
Render with:
    venv/bin/manim -qh scripts/flash_attention_calculator_analogy.py NaiveAttentionAnalogy
    venv/bin/manim -qh scripts/flash_attention_calculator_analogy.py FlashAttentionAnalogy
"""

from manim import (
    AnimationGroup,
    BOLD,
    DOWN,
    FadeIn,
    FadeOut,
    Indicate,
    LEFT,
    Line,
    RIGHT,
    ReplacementTransform,
    RoundedRectangle,
    Scene,
    SurroundingRectangle,
    Text,
    Transform,
    TransformFromCopy,
    UP,
    VGroup,
)


class MultiplicationAnalogyBase(Scene):
    BG = "#F5EFE6"
    INK = "#1F2937"
    MUTED = "#6B7280"
    PAPER = "#FFF9EE"
    PAPER_LINE = "#DCCDB6"
    SCRATCH = "#F7E173"
    CALC = "#111827"
    RED = "#C2410C"
    RED_SOFT = "#FED7AA"
    GREEN = "#0F766E"
    GREEN_SOFT = "#A7F3D0"
    HILITE = "#FDE68A"
    HILITE_EDGE = "#D97706"

    RAW_MULTIPLICAND = "12345"
    RAW_MULTIPLIER = "67891"
    ACTIVE_DIGITS = ["1", "9", "8", "7", "6"]
    PARTIAL_ROWS = ["12345", "111105", "98760", "86415", "74070"]
    CONTRIBUTIONS = ["12345", "1111050", "9876000", "86415000", "740700000"]
    RUNNING_TOTALS = ["12345", "1123395", "10999395", "97414395", "838114395"]
    RESULT = "838114395"
    TOTAL_COLS = 9

    def text_obj(self, raw, size=22, color=None, weight=BOLD, font="Helvetica Neue", disable_ligatures=False):
        return Text(
            raw,
            font=font,
            font_size=size,
            weight=weight,
            color=color or self.INK,
            disable_ligatures=disable_ligatures,
        )

    def glyph_text(self, raw, size=34, color=None):
        return self.text_obj(raw, size=size, color=color, font="Menlo", disable_ligatures=True)

    def label_text(self, text, size=22, color=None):
        return self.text_obj(text, size=size, color=color)

    def chip(self, text, fill, text_color):
        label = self.label_text(text, size=19, color=text_color)
        bg = SurroundingRectangle(
            label,
            buff=0.16,
            corner_radius=0.18,
            stroke_width=0,
            fill_color=fill,
            fill_opacity=1.0,
        )
        return VGroup(bg, label)

    def make_column_xs(self, left_x, right_x, total_cols):
        if total_cols == 1:
            return [(left_x + right_x) / 2]
        step = (right_x - left_x) / (total_cols - 1)
        return [left_x + idx * step for idx in range(total_cols)]

    def fixed_number_row(
        self,
        raw,
        *,
        column_xs,
        y,
        color=None,
        size=34,
        trailing_blanks=0,
        prefix=None,
        prefix_gap=1.25,
    ):
        color = color or self.INK
        row = VGroup()
        total_cols = len(column_xs)
        start_idx = total_cols - trailing_blanks - len(raw)
        if start_idx < 0:
            raise ValueError(f"Not enough columns for '{raw}' with shift {trailing_blanks}")

        if len(column_xs) > 1:
            col_step = column_xs[1] - column_xs[0]
        else:
            col_step = 0.48

        if prefix is not None:
            prefix_glyph = self.glyph_text(prefix, size=size, color=color)
            prefix_x = column_xs[start_idx] - prefix_gap * col_step
            prefix_glyph.move_to([prefix_x, y, 0])
            row.add(prefix_glyph)

        digit_glyphs = VGroup()
        for idx, ch in enumerate(raw):
            glyph = self.glyph_text(ch, size=size, color=color)
            glyph.move_to([column_xs[start_idx + idx], y, 0])
            digit_glyphs.add(glyph)
            row.add(glyph)

        row.digit_glyphs = digit_glyphs
        return row

    def left_align_in_card(self, mob, left_x, y):
        mob.move_to([left_x + mob.width / 2, y, 0])
        return mob

    def paper_card(self):
        box = RoundedRectangle(
            corner_radius=0.26,
            width=6.05,
            height=5.78,
            stroke_color="#000000",
            stroke_opacity=0.12,
            stroke_width=1.6,
            fill_color=self.PAPER,
            fill_opacity=1.0,
        )
        title = self.label_text("Paper", size=21)
        return {"box": box, "title": title, "group": VGroup(box, title)}

    def scratch_card(self):
        box = RoundedRectangle(
            corner_radius=0.26,
            width=4.78,
            height=2.34,
            stroke_color="#000000",
            stroke_opacity=0.12,
            stroke_width=1.6,
            fill_color=self.SCRATCH,
            fill_opacity=1.0,
        )
        title = self.label_text("Scratchpad", size=19)
        return {"box": box, "title": title, "group": VGroup(box, title)}

    def calculator(self):
        body = RoundedRectangle(
            corner_radius=0.3,
            width=2.25,
            height=2.1,
            stroke_width=1.6,
            stroke_color="#000000",
            stroke_opacity=0.12,
            fill_color=self.CALC,
            fill_opacity=1.0,
        )
        screen = RoundedRectangle(
            corner_radius=0.12,
            width=1.5,
            height=0.4,
            stroke_width=0,
            fill_color=self.GREEN_SOFT,
            fill_opacity=1.0,
        )
        screen.move_to(body.get_top() + DOWN * 0.43)

        buttons = VGroup()
        for row in range(3):
            for col in range(3):
                button = RoundedRectangle(
                    corner_radius=0.06,
                    width=0.24,
                    height=0.18,
                    stroke_width=0,
                    fill_color="#E5E7EB",
                    fill_opacity=0.96,
                )
                button.move_to(
                    body.get_bottom()
                    + UP * (0.42 + row * 0.28)
                    + LEFT * 0.44
                    + RIGHT * (col * 0.43)
                )
                buttons.add(button)

        label = self.label_text("Calculator", size=19, color=self.INK)
        label.next_to(body, DOWN, buff=0.15)
        return {
            "body": body,
            "screen": screen,
            "buttons": buttons,
            "label": label,
            "group": VGroup(body, screen, buttons, label),
        }

    def paper_row_y(self, row):
        return self.paper_grid_top_y - row * self.paper_row_gap

    def paper_text_offset(self, row):
        if row < 1.7:
            return -0.11
        if row < 2.2:
            return -0.08
        return -0.19

    def paper_number(self, raw, row, shift_digits=0, color=None, size=33, prefix=None):
        return self.fixed_number_row(
            raw,
            column_xs=self.paper_digit_xs,
            y=self.paper_row_y(row) + self.paper_text_offset(row),
            size=size,
            color=color,
            trailing_blanks=shift_digits,
            prefix=prefix,
        )

    def paper_rule(self, row):
        y = self.paper_row_y(row)
        return Line(
            [self.paper_rule_left_x, y, 0],
            [self.paper_rule_right_x, y, 0],
            stroke_color=self.INK,
            stroke_width=2.0,
            stroke_opacity=0.72,
        )

    def scratch_number(self, raw, slot, color, size=23):
        return self.fixed_number_row(
            raw,
            column_xs=self.scratch_digit_xs,
            y=self.scratch_slot_y[slot],
            size=size,
            color=color,
        )

    def scratch_label(self, text, slot):
        label = self.text_obj(text, size=17, color=self.MUTED)
        return self.left_align_in_card(label, self.scratch_label_left_x, self.scratch_slot_y[slot])

    def calc_screen_text(self, screen_box, text):
        label = self.glyph_text(text, size=18, color=self.INK)
        label.move_to(screen_box.get_center() + DOWN * 0.01)
        return label

    def layout(self, title_text, subtitle_text, chip_text, chip_fill, chip_text_color):
        self.camera.background_color = self.BG

        title = self.label_text(title_text, size=34)
        subtitle = self.text_obj(subtitle_text, size=20, color=self.MUTED, weight="NORMAL")
        heading = VGroup(title, subtitle).arrange(DOWN, buff=0.11)
        heading.to_edge(UP, buff=0.22)

        chip = self.chip(chip_text, chip_fill, chip_text_color)
        chip.next_to(heading, DOWN, buff=0.16)
        chip.move_to([0.7, chip.get_center()[1], 0])

        paper = self.paper_card()
        paper["group"].move_to(LEFT * 3.75 + DOWN * 1.04)
        paper["title"].move_to(
            [paper["box"].get_center()[0], paper["box"].get_top()[1] - 0.28, 0]
        )

        self.paper_rule_left_x = paper["box"].get_left()[0] + 0.72
        self.paper_rule_right_x = paper["box"].get_right()[0] - 0.44
        self.paper_digit_xs = self.make_column_xs(
            paper["box"].get_left()[0] + 1.65,
            paper["box"].get_right()[0] - 0.54,
            self.TOTAL_COLS,
        )
        self.paper_grid_top_y = paper["box"].get_top()[1] - 0.90
        self.paper_row_gap = 0.47

        paper_rules = VGroup()
        for idx in range(9):
            y = self.paper_row_y(idx)
            rule = Line(
                [self.paper_rule_left_x, y, 0],
                [self.paper_rule_right_x, y, 0],
                stroke_color=self.PAPER_LINE,
                stroke_width=1.0,
                stroke_opacity=0.48,
            )
            paper_rules.add(rule)
        paper["rules"] = paper_rules
        paper["group"].add(paper_rules)

        scratch = self.scratch_card()
        scratch["group"].move_to(RIGHT * 2.42 + UP * 0.17)
        scratch["title"].move_to(
            [scratch["box"].get_center()[0], scratch["box"].get_top()[1] - 0.22, 0]
        )

        self.scratch_label_left_x = scratch["box"].get_left()[0] + 0.24
        self.scratch_divider_x = scratch["box"].get_left()[0] + 1.38
        self.scratch_digit_xs = self.make_column_xs(
            self.scratch_divider_x + 0.34,
            scratch["box"].get_right()[0] - 0.34,
            self.TOTAL_COLS,
        )
        self.scratch_slot_y = {
            "partial": scratch["box"].get_center()[1] - 0.02,
            "block": scratch["box"].get_center()[1] + 0.34,
            "running": scratch["box"].get_center()[1] - 0.34,
        }
        scratch_vertical_divider = Line(
            [self.scratch_divider_x, scratch["box"].get_top()[1] - 0.56, 0],
            [self.scratch_divider_x, scratch["box"].get_bottom()[1] + 0.34, 0],
            stroke_color=self.INK,
            stroke_width=1.2,
            stroke_opacity=0.22,
        )
        scratch["vertical_divider"] = scratch_vertical_divider
        scratch["group"].add(scratch_vertical_divider)

        calc = self.calculator()
        calc["group"].move_to(RIGHT * 2.45 + DOWN * 2.08)

        a_num = self.paper_number(self.RAW_MULTIPLICAND, row=0.18, size=33)
        b_num = self.paper_number(self.RAW_MULTIPLIER, row=1.42, size=33, prefix="x")
        top_rule = self.paper_rule(2.04)

        return {
            "heading": heading,
            "chip": chip,
            "paper": paper,
            "scratch": scratch,
            "calc": calc,
            "a_num": a_num,
            "b_num": b_num,
            "b_digits": b_num.digit_glyphs,
            "top_rule": top_rule,
        }

    def active_digit_marker(self, digit_glyph, color):
        return SurroundingRectangle(
            digit_glyph,
            buff=0.06,
            corner_radius=0.09,
            stroke_width=1.2,
            stroke_color=self.HILITE_EDGE,
            stroke_opacity=0.55,
            fill_color=color,
            fill_opacity=0.32,
        )


class NaiveAttentionAnalogy(MultiplicationAnalogyBase):
    def construct(self):
        ui = self.layout(
            title_text="Naive Multiplication",
            subtitle_text="Write every intermediate product to paper",
            chip_text="many DRAM writes",
            chip_fill=self.RED_SOFT,
            chip_text_color=self.RED,
        )

        partial_label = self.scratch_label("partial", "partial")
        partial_value = self.scratch_number(self.PARTIAL_ROWS[0], "partial", color=self.RED)
        partial_value.set_opacity(0.0)
        screen_text = self.calc_screen_text(ui["calc"]["screen"], "x 1")
        active_indices = [4, 3, 2, 1, 0]
        active_marker = self.active_digit_marker(ui["b_digits"][active_indices[0]], self.HILITE)

        self.play(FadeIn(ui["heading"], shift=DOWN * 0.08), run_time=0.4)
        self.play(FadeIn(ui["chip"], shift=UP * 0.05), run_time=0.24)
        self.play(
            AnimationGroup(
                FadeIn(ui["paper"]["group"], shift=RIGHT * 0.07),
                FadeIn(ui["scratch"]["group"], shift=LEFT * 0.05),
                FadeIn(ui["calc"]["group"], shift=UP * 0.05),
                lag_ratio=0.1,
            ),
            run_time=0.7,
        )
        self.bring_to_front(ui["heading"], ui["chip"])
        self.play(
            FadeIn(VGroup(active_marker, ui["a_num"], ui["b_num"], ui["top_rule"]), shift=UP * 0.05),
            FadeIn(partial_label, shift=RIGHT * 0.03),
            FadeIn(screen_text),
            run_time=0.34,
        )

        for idx, (digit, partial, active_idx) in enumerate(
            zip(self.ACTIVE_DIGITS, self.PARTIAL_ROWS, active_indices)
        ):
            new_screen = self.calc_screen_text(ui["calc"]["screen"], f"x {digit}")
            new_partial = self.scratch_number(partial, "partial", color=self.RED)
            new_marker = self.active_digit_marker(ui["b_digits"][active_idx], self.HILITE)
            paper_row = self.paper_number(
                partial,
                row=2.42 + idx,
                shift_digits=idx,
                color=self.RED,
                size=29,
            )

            if idx == 0:
                self.play(FadeIn(partial_value), run_time=0.18)
            else:
                self.play(
                    Transform(screen_text, new_screen),
                    ReplacementTransform(partial_value, new_partial),
                    ReplacementTransform(active_marker, new_marker),
                    run_time=0.28,
                )
                partial_value = new_partial
                active_marker = new_marker

            self.play(
                Indicate(ui["calc"]["screen"], color=self.GREEN_SOFT, scale_factor=1.01),
                TransformFromCopy(partial_value, paper_row),
                run_time=0.31,
            )

        bottom_rule = self.paper_rule(7.42)
        final_result = self.paper_number(self.RESULT, row=8.02, color=self.RED, size=33)

        self.play(
            FadeIn(bottom_rule, shift=UP * 0.02),
            FadeIn(final_result, shift=UP * 0.02),
            run_time=0.3,
        )
        self.play(Indicate(final_result, color=self.RED_SOFT, scale_factor=1.01), run_time=0.28)
        self.play(FadeOut(active_marker), run_time=0.18)
        self.wait(0.8)


class FlashAttentionAnalogy(MultiplicationAnalogyBase):
    def construct(self):
        ui = self.layout(
            title_text="Smart Multiplication",
            subtitle_text="Keep a running sum on the scratchpad",
            chip_text="one final DRAM write",
            chip_fill=self.GREEN_SOFT,
            chip_text_color=self.GREEN,
        )

        partial_label = self.scratch_label("block", "block")
        running_label = self.scratch_label("running", "running")
        row_divider = Line(
            [ui["scratch"]["box"].get_left()[0] + 0.28, 0, 0],
            [ui["scratch"]["box"].get_right()[0] - 0.28, 0, 0],
            stroke_color=self.INK,
            stroke_width=1.2,
            stroke_opacity=0.18,
        )
        row_divider.move_to(
            [
                ui["scratch"]["box"].get_center()[0],
                (self.scratch_slot_y["block"] + self.scratch_slot_y["running"]) / 2,
                0,
            ]
        )

        partial_value = self.scratch_number(self.CONTRIBUTIONS[0], "block", color=self.GREEN, size=22)
        running_value = self.scratch_number(self.RUNNING_TOTALS[0], "running", color=self.GREEN, size=22)
        screen_text = self.calc_screen_text(ui["calc"]["screen"], "x 1")
        active_indices = [4, 3, 2, 1, 0]
        active_marker = self.active_digit_marker(ui["b_digits"][active_indices[0]], self.HILITE)

        self.play(FadeIn(ui["heading"], shift=DOWN * 0.08), run_time=0.4)
        self.play(FadeIn(ui["chip"], shift=UP * 0.05), run_time=0.24)
        self.play(
            AnimationGroup(
                FadeIn(ui["paper"]["group"], shift=RIGHT * 0.07),
                FadeIn(ui["scratch"]["group"], shift=LEFT * 0.05),
                FadeIn(ui["calc"]["group"], shift=UP * 0.05),
                lag_ratio=0.1,
            ),
            run_time=0.7,
        )
        self.bring_to_front(ui["heading"], ui["chip"])
        self.play(
            FadeIn(VGroup(active_marker, ui["a_num"], ui["b_num"], ui["top_rule"]), shift=UP * 0.05),
            FadeIn(VGroup(partial_label, running_label, row_divider), shift=RIGHT * 0.03),
            FadeIn(VGroup(partial_value, running_value)),
            FadeIn(screen_text),
            run_time=0.36,
        )

        for digit, contrib, total, active_idx in zip(
            self.ACTIVE_DIGITS[1:],
            self.CONTRIBUTIONS[1:],
            self.RUNNING_TOTALS[1:],
            active_indices[1:],
        ):
            new_screen = self.calc_screen_text(ui["calc"]["screen"], f"x {digit}")
            new_partial = self.scratch_number(contrib, "block", color=self.GREEN, size=22)
            new_running = self.scratch_number(total, "running", color=self.GREEN, size=22)
            new_marker = self.active_digit_marker(ui["b_digits"][active_idx], self.HILITE)

            self.play(
                Transform(screen_text, new_screen),
                ReplacementTransform(partial_value, new_partial),
                ReplacementTransform(running_value, new_running),
                ReplacementTransform(active_marker, new_marker),
                run_time=0.55,
            )
            partial_value = new_partial
            running_value = new_running
            active_marker = new_marker
            self.play(Indicate(running_value, color=self.GREEN_SOFT, scale_factor=1.012), run_time=0.18)

        final_result = self.paper_number(self.RESULT, row=2.52, color=self.GREEN, size=33)

        self.play(TransformFromCopy(running_value, final_result), run_time=0.56)
        self.play(Indicate(final_result, color=self.GREEN_SOFT, scale_factor=1.012), run_time=0.3)
        self.play(FadeOut(active_marker), run_time=0.18)
        self.wait(0.8)
