import polars as pl

_instructions_tabular_filepath = "backend/src/ai/files/Streetsmart Walkability Assessment Table.xlsx"

def get_instructions():
    criteria = """ACCESSIBILITY (wide pavements, pavements on both sides, ramps, raised crosswalks, signage).
SAFETY(active edges providing eyes on the street, speed reduction measures, clearly marked crosswalks, crosswalks with pinch points, intersection design, low speed limits, appropriate lighting for pedestrians, presence of active mobility lanes in high speed avenues, otherwise low volumes of cars).
COMFORT (shade/ urban cooling, public seating, noise level, materials, sheltered bus stations where present).
BEAUTY (active edges, parking, play elements, landscape, materials, street furniture, architecture).
ECOLOGY (trees, raingardens/ bioswales, native vegetation not requiring excessive irrigation, live hedges)."""

    instr = """You will rank the walkability of the two images based on the following criteria:
{0}
Please format your response as 
{{
    "description1": "text up to 250 characters",
    "description2": "text up to 250 characters",
    "walk_surface1": {{"pavement", "grass", "clay", "asphalt", "dirt", "another option of your choice"}},
    "walk_surface2": {{"pavement", "grass", "clay", "asphalt", "dirt", "another option of your choice"}},
    "car_prevalence1": {{"none", "low", "medium", "high"}},
    "car_prevalence2": {{"none", "low", "medium", "high"}},
    "reasoning": "text up to 250 characters describing why you picked the first or second option",
    "best": {{1 or 2}},
    "confidence": decimal in [0,1]
}}
    """.format(
        criteria
    )

    return instr

if __name__ == "__main__":
    print(get_instructions())
    