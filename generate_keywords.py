#!/usr/bin/env python3
"""
Script to generate keyword database from resume dataset.
Run this once to create the position_keywords.pkl file.
"""

from job_keyword_model import generate_keyword_database, save_keyword_database

def main():
    print("Generating keyword database from Resume.csv...")
    
    try:
        # Generate keyword database
        keyword_db = generate_keyword_database("Resume.csv")
        
        # Save to pickle file
        save_keyword_database(keyword_db, "position_keywords.pkl")
        
        print(f"✅ Successfully generated keyword database with {len(keyword_db)} job categories:")
        for category in keyword_db.keys():
            print(f"  - {category}")
        
        print("\n📁 Saved as 'position_keywords.pkl'")
        
    except FileNotFoundError:
        print("❌ Error: Resume.csv file not found.")
        print("Please ensure Resume.csv is in the same directory.")
    except Exception as e:
        print(f"❌ Error generating keyword database: {e}")

if __name__ == "__main__":
    main()