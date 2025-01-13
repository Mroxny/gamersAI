from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import sqlite3
import os
import dtos

DB_PATH = "data/01_raw/games.db"

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def get_game_by_name(name: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT * FROM games WHERE name = ?;
        """
    cursor.execute(query, (name,))
    result = cursor.fetchone()
    conn.close()
    return result

def get_all_games(limit: int, offset: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT * FROM games LIMIT ? OFFSET ?;
        """
    cursor.execute(query, (limit, offset))
    results = cursor.fetchall()
    conn.close()
    return results

def add_game(game: dtos.GameDTO):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        INSERT INTO games (
            Name, Release date, Estimated owners, Peak CCU, Required age, Price, DLC count,
            About the game, Supported languages, Full audio languages, Reviews, Header image,
            Website, Support url, Support email, Windows, Mac, Linux, Metacritic score, Metacritic url,
            User score, Positive, Negative, Score rank, Achievements, Recommendations, Notes,
            Average playtime forever, Average playtime two weeks, Median playtime forever, Median playtime two weeks,
            Developers, Publishers, Categories, Genres, Tags, Screenshots, Movies
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
    cursor.execute(query,(
                game.Name, game.Release_date, game.Estimated_owners, game.Peak_CCU, game.Required_age,
                game.Price, game.DLC_count, game.About_the_game, game.Supported_languages, game.Full_audio_languages,
                game.Reviews, game.Header_image, game.Website, game.Support_url, game.Support_email, game.Windows,
                game.Mac, game.Linux, game.Metacritic_score, game.Metacritic_url, game.User_score, game.Positive,
                game.Negative, game.Score_rank, game.Achievements, game.Recommendations, game.Notes,
                game.Average_playtime_forever, game.Average_playtime_two_weeks, game.Median_playtime_forever,
                game.Median_playtime_two_weeks, game.Developers, game.Publishers, game.Categories, game.Genres,
                game.Tags, game.Screenshots, game.Movies
            ))
    conn.commit()
    conn.close()