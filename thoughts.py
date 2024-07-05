from typing import List

from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


class Ingredient(BaseModel):
    name: str = Field(description="Name of the ingredient")
    quantity: str = Field(description="Quantity of the ingredient in gram")


class Menu(BaseModel):
    day: int = Field(description="Number of the day of the menu")
    meals: str = Field(description="Meals of the menu")
    title: str = Field(description="Recipe name")
    ingredients: List[Ingredient]
    steps: str = Field(description="Instructions for making the recipe")
    preparation_time: str = Field(
        description="Preparation time for the recipe in minutes")
    cooking_time: str = Field(
        description="Cooking time for the recipe in minutes")
    calories: str = Field(description="Calories of the recipe")
    nutritional_value: str = Field(
        description="Nutritional value for the recipe")


class Diet():
    menu: List[Menu]


def menu(diet, n_days, n_meals, language):
    prompt = """
    You are an assistant specialized in {diet} diets.
    Generate a menu for {n_days} days, with {n_meals} meals per day. All the test should be done in {language}.
    The recipes cannot be repeated, they must be varied and simple.
    You must be detailed both in the quantities of the ingredients and in the steps to follow for the recipe.
    It is not valid to complete the text with something like, repeat this the rest of the days. It is necessary that you declare all recipes.
    """
    parser = PydanticOutputParser(pydantic_object=Diet)
    template = PromptTemplate(
        input_variables=["diet"],
        template=prompt + "Use the following format: \n {parser}",
        partial_variables={"parser": parser}
    )
    llm = Ollama(temperature=0.5, model="gemma2")
    chain = template | llm
    return chain.invoke({"diet": diet, "n_days": n_days, "n_meals": n_meals, "language": language})
