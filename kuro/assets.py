import base64
import cairosvg
from typing import Dict

# -----------------------------
# Embedded Assets (Generated)
# -----------------------------
def generate_avatar_svg(mood: str = 'neutral', size: int = 256) -> str:
    """Generates a dynamic SVG avatar for Kuro."""
    # Colors
    skin = "#FBEFE1"
    hair = "#2c2c2c"
    hair_highlight = "#4a4a4a"
    eye_white = "#FFFFFF"
    iris = "#e06c75" # Kuro's theme color
    outline = "#1a1a1a"
    blush = "rgba(224, 108, 117, 0.5)"

    # Base structure
    svg = f'<svg width="{size}" height="{size}" viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg">'
    svg += f'<rect width="256" height="256" fill="none"/>' # Transparent background

    # Hair (Back) - Twin-tails
    svg += f'<path d="M 60 140 Q 20 220 70 250 T 50 160" fill="{hair}"/>'
    svg += f'<path d="M 196 140 Q 236 220 186 250 T 206 160" fill="{hair}"/>'

    # Face
    svg += f'<circle cx="128" cy="128" r="80" fill="{skin}" stroke="{outline}" stroke-width="3"/>'

    # Hair (Front)
    svg += f'<path d="M 48,90 C 48,40 208,40 208,90 Q 128,70 48,90 Z" fill="{hair}"/>'
    # Hair highlight
    svg += f'<path d="M 80,60 C 90,50 160,50 170,60 Q 128,55 80,60 Z" fill="{hair_highlight}"/>'


    # Expressions
    eye_y = 120
    # Default: Neutral
    left_eye = f'<circle cx="95" cy="{eye_y}" r="14" fill="{eye_white}" stroke="{outline}" stroke-width="2"/>'
    left_eye += f'<circle cx="95" cy="{eye_y}" r="8" fill="{iris}"/>'
    left_eye += f'<circle cx="98" cy="{eye_y-3}" r="3" fill="{eye_white}"/>' # highlight
    right_eye = f'<circle cx="161" cy="{eye_y}" r="14" fill="{eye_white}" stroke="{outline}" stroke-width="2"/>'
    right_eye += f'<circle cx="161" cy="{eye_y}" r="8" fill="{iris}"/>'
    right_eye += f'<circle cx="164" cy="{eye_y-3}" r="3" fill="{eye_white}"/>' # highlight
    mouth = f'<path d="M 115 170 Q 128 175 141 170" stroke="{outline}" stroke-width="2" fill="none"/>'
    blush_l = ''
    blush_r = ''

    if mood == 'playful':
        # Wink ;)
        left_eye = f'<path d="M 85 {eye_y-5} Q 95 {eye_y} 105 {eye_y-5}" stroke="{outline}" stroke-width="2.5" fill="none"/>'
        mouth = f'<path d="M 115 165 Q 128 180 141 165" stroke="{outline}" stroke-width="2" fill="none"/>'
    elif mood == 'jealous':
        # Angry eyes
        left_eye = f'<path d="M 80 {eye_y-10} L 110 {eye_y-2}" stroke="{outline}" stroke-width="3" fill="none"/>'
        left_eye += f'<path d="M 80 {eye_y+2} L 110 {eye_y+10}" stroke="{outline}" stroke-width="3" fill="none" transform="rotate(5 95 {eye_y})"/>'
        right_eye = f'<path d="M 146 {eye_y-2} L 176 {eye_y-10}" stroke="{outline}" stroke-width="3" fill="none"/>'
        right_eye += f'<path d="M 146 {eye_y+10} L 176 {eye_y+2}" stroke="{outline}" stroke-width="3" fill="none" transform="rotate(-5 161 {eye_y})"/>'
        mouth = f'<path d="M 115 175 Q 128 165 141 175" stroke="{outline}" stroke-width="2" fill="none"/>'
        blush_l = f'<ellipse cx="90" cy="145" rx="20" ry="8" fill="{blush}"/>'
        blush_r = f'<ellipse cx="166" cy="145" rx="20" ry="8" fill="{blush}"/>'
    elif mood == 'scheming':
        # Sly, half-closed eyes
        left_eye = f'<path d="M 85 {eye_y-5} Q 95 {eye_y-10} 105 {eye_y-5}" stroke="{outline}" stroke-width="2.5" fill="none"/>'
        left_eye += f'<path d="M 85 {eye_y+5} Q 95 {eye_y} 105 {eye_y+5}" stroke="{outline}" stroke-width="2.5" fill="none"/>'
        right_eye = f'<path d="M 151 {eye_y-5} Q 161 {eye_y-10} 171 {eye_y-5}" stroke="{outline}" stroke-width="2.5" fill="none"/>'
        right_eye += f'<path d="M 151 {eye_y+5} Q 161 {eye_y} 171 {eye_y+5}" stroke="{outline}" stroke-width="2.5" fill="none"/>'
        mouth = f'<path d="M 115 165 C 120 175, 136 175, 141 165" stroke="{outline}" stroke-width="2" fill="none"/>' # Smirk
    elif mood == 'thoughtful':
        # Eyes looking sideways
        left_eye = f'<circle cx="95" cy="{eye_y}" r="14" fill="{eye_white}" stroke="{outline}" stroke-width="2"/>'
        left_eye += f'<circle cx="100" cy="{eye_y+2}" r="7" fill="{iris}"/>'
        right_eye = f'<circle cx="161" cy="{eye_y}" r="14" fill="{eye_white}" stroke="{outline}" stroke-width="2"/>'
        right_eye += f'<circle cx="166" cy="{eye_y+2}" r="7" fill="{iris}"/>'
        mouth = f'<line x1="118" y1="170" x2="138" y2="170" stroke="{outline}" stroke-width="2"/>'
    elif mood == 'curious':
        # Wide, interested eyes
        left_eye = f'<circle cx="95" cy="{eye_y}" r="16" fill="{eye_white}" stroke="{outline}" stroke-width="2"/>' # Slightly larger
        left_eye += f'<circle cx="95" cy="{eye_y}" r="9" fill="{iris}"/>' # Larger iris
        left_eye += f'<circle cx="99" cy="{eye_y-4}" r="4" fill="{eye_white}"/>' # Bigger highlight
        right_eye = f'<circle cx="161" cy="{eye_y}" r="16" fill="{eye_white}" stroke="{outline}" stroke-width="2"/>'
        right_eye += f'<circle cx="161" cy="{eye_y}" r="9" fill="{iris}"/>'
        right_eye += f'<circle cx="165" cy="{eye_y-4}" r="4" fill="{eye_white}"/>'
        mouth = f'<path d="M 120 170 Q 128 172 136 170" stroke="{outline}" stroke-width="2" fill="none"/>' # Slightly open


    svg += blush_l + blush_r
    svg += left_eye
    svg += right_eye
    svg += mouth
    svg += '</svg>'
    return svg

def create_avatar_data() -> Dict[str, str]:
    """Generates all avatar images and returns a dict of base64 strings."""
    if not cairosvg:
        print("Warning: cairosvg is not installed. Avatars will be disabled.")
        return {mood: "" for mood in ['neutral', 'playful', 'jealous', 'scheming', 'thoughtful', 'curious']}

    avatars = {}
    for mood in ['neutral', 'playful', 'jealous', 'scheming', 'thoughtful', 'curious']:
        try:
            svg_text = generate_avatar_svg(mood)
            png_bytes = cairosvg.svg2png(bytestring=svg_text.encode('utf-8'))
            b64_string = base64.b64encode(png_bytes).decode('utf-8')
            avatars[mood] = b64_string
        except Exception as e:
            print(f"Error generating avatar for mood '{mood}': {e}")
            avatars[mood] = "" # fallback to empty
    return avatars

AVATAR_DATA = create_avatar_data()
