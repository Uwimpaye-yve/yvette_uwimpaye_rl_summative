import math

import pygame

W, H = 1150, 640
FPS = 30

BG = (232, 240, 228)
ROAD = (210, 204, 192)
PANEL = (255, 255, 255)
TEXT = (40, 45, 50)
MUTED = (115, 120, 125)
UNKNOWN = (185, 190, 195)
CHW_BLUE = (33, 99, 214)
GOOD = (46, 160, 67)
BAD = (210, 50, 60)
RISK_COLORS = {0: (76, 175, 80), 1: (255, 193, 7), 2: (255, 138, 0), 3: (220, 53, 69)}
LABELS = {0: "Well", 1: "Mild", 2: "Moderate", 3: "Crisis"}
SKIN = (120, 80, 55)
SHIRTS = [(90, 120, 200), (200, 90, 140), (80, 160, 150), (150, 110, 200), (220, 130, 70)]
MISSION = "Mission: keep 5 youth safe for 20 weeks. One action per week: visit to find out, or refer to the clinic (2 beds)."


def action_name(a):
    if a is None:
        return "Start"
    if a < 5:
        return f"Visit & screen youth {a + 1}"
    if a < 10:
        return f"Refer youth {a - 4} to health centre"
    return "Group session at school"


def lighter(color, amount=0.5):
    return tuple(int(c + (255 - c) * amount) for c in color)


def ip(point):
    return (int(point[0]), int(point[1]))


class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("CHW Triage & Referral - Kigali")
        self.f_small = pygame.font.SysFont("segoeui,arial", 15)
        self.f_med = pygame.font.SysFont("segoeui,arial", 18)
        self.f_bold = pygame.font.SysFont("segoeui,arial", 19, bold=True)
        self.f_big = pygame.font.SysFont("segoeui,arial", 34, bold=True)
        self.clock = pygame.time.Clock()
        self.speed = 1.0
        self.paused = False
        self.houses = [(40 + i * 150, 330) for i in range(5)]
        self.new_episode()

    # ---------- episode bookkeeping ----------
    def new_episode(self):
        self.total = 0.0
        self.log = []
        self.screened = [False] * 5
        self.prev_referred = [0] * 5
        self.correct_refs = 0
        self.wrong_refs = 0
        self.pos = (400, 283)
        self.event = "New caseload: nobody has been screened yet"
        self.say = "Who needs help this week?"

    def target_pos(self, a):
        if a is None:
            return (400, 283)
        if a < 5:
            return (self.houses[a][0] + 60, 283)
        if a < 10:
            return (655, 283)
        return (140, 283)

    # ---------- main entry, called by the environment after every step ----------
    def draw(self, env, reward=0.0):
        if env.week == 0:
            self.new_episode()
            self.animate(env, reward=None, frames=1, hold=20)
            return

        a = env.last_action
        reward = float(reward)
        self.total += reward
        if a < 5:
            self.say = "How are you feeling?"
            self.screened[a] = True
            self.event = f"Week {env.week}: screened Youth {a + 1} - {LABELS[int(env.last_score[a])].lower()}"
        elif a < 10:
            i = a - 5
            self.say = f"Referral for Youth {i + 1}"
            if self.prev_referred[i] == 1:
                self.event = f"Week {env.week}: Youth {i + 1} is already at the clinic"
            elif sum(self.prev_referred) >= 2:
                self.event = f"Week {env.week}: clinic full - Youth {i + 1} could not be admitted"
            elif env.referred[i] == 1:
                self.correct_refs += 1
                self.event = f"Week {env.week}: referred Youth {i + 1} - needed care, good call!"
            else:
                self.wrong_refs += 1
                self.event = f"Week {env.week}: referred Youth {i + 1} - referral was not needed"
        else:
            self.say = "Group session today!"
            self.event = f"Week {env.week}: group session at school"

        short = action_name(a).replace(" to health centre", "").replace("Visit & screen", "Screen")
        self.log.append((env.week, short, reward))
        self.log = self.log[-7:]

        self.animate(env, reward, frames=12, hold=18)
        self.prev_referred = [int(x) for x in env.referred]

        crisis = sum(1 for i in range(5) if env.risk[i] == 3 and env.referred[i] == 0)
        if env.week >= 20 or crisis >= 3:
            self.summary(env, crisis >= 3)

    # ---------- animation loop ----------
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_UP:
                    self.speed = min(self.speed * 1.5, 6.0)
                elif event.key == pygame.K_DOWN:
                    self.speed = max(self.speed / 1.5, 0.25)

    def animate(self, env, reward, frames, hold):
        start, end = self.pos, self.target_pos(env.last_action)
        frames = max(1, int(frames / self.speed))
        hold = max(1, int(hold / self.speed))
        for f in range(frames + hold):
            self.handle_events()
            while self.paused:
                self.render(env, self.pos, reward, 1.0, 0.0, self.say)
                self.handle_events()
                self.clock.tick(FPS)
            t = min(1.0, (f + 1) / frames)
            moving = t < 1.0 and start != end
            t = t * t * (3 - 2 * t)  # smooth start and stop
            self.pos = (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)
            phase = f * 0.9 if moving else 0.0
            self.render(env, self.pos, reward, (f + 1) / (frames + hold), phase, None if moving else self.say)
            self.clock.tick(FPS)

    # ---------- drawing ----------
    def text(self, msg, x, y, font, color=TEXT, center=False):
        surf = font.render(msg, True, color)
        if center:
            x -= surf.get_width() // 2
        self.screen.blit(surf, (int(x), int(y)))

    def human(self, x, y, k, shirt, pants=(50, 55, 70), phase=0.0, mood=None, chw=False, backpack=False):
        """Draw a little person standing at (x, y) = feet. phase animates walking, mood draws a face."""
        s = self.screen
        swing = math.sin(phase) * 0.6
        if backpack:
            pygame.draw.rect(s, (110, 70, 45), (int(x - 13 * k), int(y - 44 * k), int(10 * k), int(18 * k)), border_radius=3)
        for side in (1, -1):  # legs
            a = swing * side
            foot = (x + math.sin(a) * 20 * k, y - 22 * k + math.cos(a) * 20 * k)
            pygame.draw.line(s, pants, ip((x, y - 22 * k)), ip(foot), max(2, int(5 * k)))
            pygame.draw.circle(s, (35, 35, 35), ip(foot), max(2, int(3 * k)))
        pygame.draw.rect(s, shirt, (int(x - 9 * k), int(y - 46 * k), int(18 * k), int(26 * k)), border_radius=int(6 * k))
        if chw:  # white badge with a red cross + shoulder bag
            pygame.draw.rect(s, (255, 255, 255), (int(x - 5 * k), int(y - 42 * k), int(10 * k), int(10 * k)), border_radius=2)
            pygame.draw.rect(s, (200, 50, 50), (int(x - 1 * k), int(y - 41 * k), max(1, int(2 * k)), int(8 * k)))
            pygame.draw.rect(s, (200, 50, 50), (int(x - 4 * k), int(y - 38 * k), int(8 * k), max(1, int(2 * k))))
            pygame.draw.rect(s, (140, 90, 50), (int(x + 7 * k), int(y - 30 * k), int(9 * k), int(10 * k)), border_radius=3)
        for side in (1, -1):  # arms
            a = -swing * side
            hand = (x + side * 10 * k + math.sin(a) * 15 * k, y - 42 * k + math.cos(a) * 15 * k)
            pygame.draw.line(s, shirt, ip((x + side * 7 * k, y - 42 * k)), ip(hand), max(2, int(4 * k)))
            pygame.draw.circle(s, SKIN, ip(hand), max(2, int(3 * k)))
        pygame.draw.circle(s, SKIN, ip((x, y - 56 * k)), int(9 * k))  # head
        if chw:
            pygame.draw.ellipse(s, CHW_BLUE, (int(x - 10 * k), int(y - 67 * k), int(20 * k), int(8 * k)))  # cap
            pygame.draw.rect(s, CHW_BLUE, (int(x), int(y - 62 * k), int(13 * k), int(3 * k)))
        else:
            pygame.draw.ellipse(s, (30, 25, 20), (int(x - 9 * k), int(y - 66 * k), int(18 * k), int(8 * k)))  # hair
        for side in (-1, 1):  # eyes
            pygame.draw.circle(s, (20, 20, 20), ip((x + side * 3 * k, y - 57 * k)), max(1, int(1.4 * k)))
        mouth = (int(x - 4 * k), int(y - 55 * k), int(8 * k), int(6 * k))
        if mood is None or mood == 0:
            pygame.draw.arc(s, (20, 20, 20), mouth, math.pi, 2 * math.pi, max(1, int(1.5 * k)))  # smile
        elif mood == 1:
            pygame.draw.line(s, (20, 20, 20), ip((x - 3 * k, y - 52 * k)), ip((x + 3 * k, y - 52 * k)), max(1, int(1.5 * k)))
        else:
            sad = (int(x - 4 * k), int(y - 52 * k), int(8 * k), int(6 * k))
            pygame.draw.arc(s, (20, 20, 20), sad, 0, math.pi, max(1, int(1.5 * k)))  # frown
            if mood == 3:
                pygame.draw.circle(s, (80, 150, 255), ip((x + 4 * k, y - 53 * k)), max(2, int(2 * k)))  # tear

    def bubble(self, msg, x, y):
        surf = self.f_small.render(msg, True, TEXT)
        w, h = surf.get_width() + 16, 26
        pygame.draw.rect(self.screen, (255, 255, 255), (int(x), int(y), w, h), border_radius=10)
        pygame.draw.rect(self.screen, (180, 185, 190), (int(x), int(y), w, h), 1, border_radius=10)
        pygame.draw.polygon(self.screen, (255, 255, 255), [ip((x + 10, y + h - 1)), ip((x + 20, y + h - 1)), ip((x + 4, y + h + 9))])
        self.screen.blit(surf, (int(x + 8), int(y + 4)))

    def render(self, env, chw_pos, reward, progress, phase=0.0, say=None):
        s = self.screen
        s.fill(BG)

        # mission + event line
        pygame.draw.rect(s, PANEL, (20, 12, 780, 40), border_radius=10)
        self.text(MISSION, 32, 22, self.f_small)
        self.text(self.event, 32, 62, self.f_bold)

        # road
        pygame.draw.rect(s, ROAD, (30, 270, 760, 26), border_radius=13)
        for x in range(50, 780, 40):
            pygame.draw.rect(s, (240, 238, 230), (x, 281, 18, 4))

        # school
        pygame.draw.rect(s, (170, 200, 255), (40, 120, 200, 110), border_radius=12)
        self.text("School", 140, 135, self.f_bold, center=True)
        self.text("group sessions", 140, 165, self.f_small, MUTED, center=True)

        # health centre with 2 beds
        pygame.draw.rect(s, PANEL, (540, 105, 230, 125), border_radius=12)
        pygame.draw.rect(s, (200, 60, 60), (548, 113, 26, 8))
        pygame.draw.rect(s, (200, 60, 60), (557, 104, 8, 26))
        self.text("Health Centre", 665, 112, self.f_bold, center=True)
        patients = [i for i in range(5) if env.referred[i] == 1]
        for b in range(2):
            bx = 565 + b * 105
            pygame.draw.rect(s, (225, 232, 240), (bx, 150, 90, 60), border_radius=8)
            if b < len(patients):
                p = patients[b]
                pygame.draw.rect(s, (255, 255, 255), (bx + 8, 190, 74, 12), border_radius=4)  # bed
                self.human(bx + 28, 200, 0.6, SHIRTS[p], mood=int(env.risk[p]))
                self.text(f"Y{p + 1}", bx + 55, 160, self.f_med)
            else:
                self.text("free", bx + 45, 170, self.f_small, MUTED, center=True)

        # houses: big colour = what the CHW knows, small dot = true hidden state
        for i, (x, y) in enumerate(self.houses):
            true_risk = int(env.risk[i])
            known = self.screened[i] or env.last_score[i] > 0
            body = RISK_COLORS[int(env.last_score[i])] if known else UNKNOWN
            if known and env.weeks_since[i] >= 5:
                body = lighter(body, 0.45)  # old information fades
            if true_risk == 3 and env.referred[i] == 0:
                pygame.draw.rect(s, BAD, (x - 6, y - 36, 132, 132), 4, border_radius=10)
            pygame.draw.polygon(s, (120, 70, 40), [(x - 8, y), (x + 60, y - 30), (x + 128, y)])
            pygame.draw.rect(s, body, (x, y, 120, 90), border_radius=8)
            pygame.draw.rect(s, (110, 75, 50), (x + 8, y + 26, 36, 64), border_radius=4)  # door
            if env.referred[i] == 0:
                self.human(x + 26, y + 88, 0.8, SHIRTS[i], mood=true_risk, backpack=True)
            if known:
                self.text(LABELS[int(env.last_score[i])], x + 84, y + 35, self.f_med, center=True)
            else:
                self.text("?", x + 84, y + 22, self.f_big, (255, 255, 255), center=True)
            pygame.draw.circle(s, (255, 255, 255), (x + 110, y + 10), 12)
            pygame.draw.circle(s, RISK_COLORS[true_risk], (x + 110, y + 10), 9)
            if env.referred[i] == 1:
                pygame.draw.rect(s, CHW_BLUE, (x + 18, y - 22, 84, 20), border_radius=6)
                self.text("AT CLINIC", x + 60, y - 21, self.f_small, (255, 255, 255), center=True)
            self.text(f"Youth {i + 1}", x + 60, y + 96, self.f_bold, center=True)
            seen = "never screened" if not known else f"seen {int(min(env.weeks_since[i], 10))} wk ago"
            self.text(seen, x + 60, y + 120, self.f_small, MUTED, center=True)

        # legend
        lx, ly = 30, 590
        pygame.draw.rect(s, UNKNOWN, (lx, ly, 18, 18), border_radius=4)
        self.text("not screened", lx + 24, ly - 1, self.f_small)
        for k in range(4):
            pygame.draw.rect(s, RISK_COLORS[k], (lx + 130 + k * 95, ly, 18, 18), border_radius=4)
            self.text(LABELS[k], lx + 154 + k * 95, ly - 1, self.f_small)
        self.text("House colour = what the CHW knows   |   dot + face = how the youth really feels (hidden from the agent)",
                  lx, ly + 24, self.f_small, MUTED)

        # CHW
        self.human(chw_pos[0], chw_pos[1] + 8, 1.15, CHW_BLUE, phase=phase, chw=True)
        if say:
            self.bubble(say, chw_pos[0] + 14, chw_pos[1] - 108)

        # floating reward
        if reward is not None and progress < 0.95:
            color = GOOD if reward > 0 else BAD
            self.text(f"{reward:+.1f}", chw_pos[0] - 40, chw_pos[1] - 80 - 30 * progress, self.f_bold, color, center=True)

        self.side_panel(env)
        pygame.display.flip()

    def side_panel(self, env):
        s = self.screen
        px = 820
        pygame.draw.rect(s, PANEL, (px, 12, 310, 616), border_radius=12)
        self.text(f"Week {env.week} / 20", px + 20, 25, self.f_bold)
        pygame.draw.rect(s, (225, 230, 235), (px + 20, 55, 270, 12), border_radius=6)
        pygame.draw.rect(s, CHW_BLUE, (px + 20, 55, int(270 * env.week / 20), 12), border_radius=6)

        self.text("Total reward", px + 20, 85, self.f_med, MUTED)
        self.text(f"{self.total:.1f}", px + 20, 108, self.f_big, GOOD if self.total >= 0 else BAD)

        crisis = sum(1 for i in range(5) if env.risk[i] == 3 and env.referred[i] == 0)
        self.text(f"Youth in crisis: {crisis}", px + 20, 165, self.f_med, BAD if crisis else TEXT)
        self.text(f"Clinic beds used: {int(env.referred.sum())} / 2", px + 20, 192, self.f_med)
        self.text(f"Good referrals: {self.correct_refs}   Not needed: {self.wrong_refs}", px + 20, 219, self.f_small)

        self.text("Recent actions", px + 20, 255, self.f_bold)
        for k, (wk, name, r) in enumerate(reversed(self.log)):
            y = 285 + k * 30
            self.text(f"W{wk}", px + 20, y, self.f_small, MUTED)
            self.text(name, px + 60, y, self.f_small)
            self.text(f"{r:+.1f}", px + 250, y, self.f_small, GOOD if r > 0 else BAD)

        self.text("SPACE pause   UP/DOWN speed   ESC quit", px + 20, 575, self.f_small, MUTED)
        self.text(f"Speed x{self.speed:.1f}" + ("   PAUSED" if self.paused else ""), px + 20, 597, self.f_small, MUTED)

    def summary(self, env, failed):
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))
        pygame.draw.rect(self.screen, PANEL, (300, 190, 450, 230), border_radius=16)
        title = "Caseload out of control" if failed else "Episode complete"
        self.text(title, 525, 210, self.f_big, BAD if failed else GOOD, center=True)
        self.text(f"Total reward: {self.total:.1f}", 525, 270, self.f_bold, center=True)
        self.text(f"Good referrals: {self.correct_refs}    Not needed: {self.wrong_refs}", 525, 305, self.f_med, center=True)
        self.text(f"Weeks completed: {env.week} / 20", 525, 335, self.f_med, center=True)
        self.text("Next caseload starting...", 525, 380, self.f_small, MUTED, center=True)
        pygame.display.flip()
        for _ in range(max(1, int(75 / self.speed))):
            self.handle_events()
            self.clock.tick(FPS)

    def save(self, path):
        pygame.image.save(self.screen, path)

    def close(self):
        pygame.quit()