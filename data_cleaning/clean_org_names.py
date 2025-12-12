import pandas as pd
from pymongo import MongoClient
from rapidfuzz import process, fuzz
from datetime import datetime, timezone
from dotenv import load_dotenv
import os
import re
import logging

# ==== 0. Setup logging ====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==== 1. Load environment variables ====
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("DB_NAME")
collection_name = os.getenv("COLLECTION_NAME")
clear_collection = os.getenv("CLEAR_COLLECTION", "False").lower() == "true"

# ==== 2. Load Excel Files ====
try:
    all_data = pd.read_excel("all_data.xlsx")
except FileNotFoundError as e:
    logging.error(f"Excel file not found: {e}")
    exit(1)

# Define column names
org_col = "Organization/Law Firm Name"
reg_col = "Reg Code"

# Define correct names (516 law firms)
correct_names = [
    "Fish & Richardson P.C.", "Sughrue Mion, PLLC", "Foley & Lardner LLP", "Oblon, McClelland, Maier & Neustadt, L.L.P.",
    "Kilpatrick Townsend & Stockton LLP", "Birch, Stewart, Kolasch & Birch, LLP", "Harness, Dickey & Pierce, P.L.C.",
    "Schwegman Lundberg & Woessner, P.A.", "CANTOR COLBURN LLP", "Knobbe, Martens, Olson & Bear, LLP",
    "JCIPRNET(Jianq Chyun Intellectual Property Group)", "Morgan, Lewis & Bockius LLP", "Oliff PLC", "Perkins Coie LLP",
    "Slater Matsil, LLP", "Womble Bond Dickinson (US) LLP", "XSENSUS LLP", "Nixon & Vanderhye P.C.", "Dority & Manning, P.A.",
    "Seed IP Law Group LLP", "Banner & Witcoff, Ltd.", "Muncy, Geissler, Olds & Lowe, P.C.", "HAUPTMAN HAM, LLP",
    "Crowell & Moring LLP", "Polsinelli PC", "Merchant & Gould P.C.", "Leydig, Voit & Mayer, Ltd.", "Studebaker & Brackett PC",
    "Michael Best & Friedrich LLP", "Patterson + Sheridan, LLP", "ArentFox Schiff LLP", "Sterne, Kessler, Goldstein & Fox P.L.L.C.",
    "Workman Nydegger", "Holland & Hart LLP", "Greenberg Traurig, LLP", "Venable LLP", "Osha Bergman Watanabe & Burton LLP",
    "Haynes and Boone, LLP", "K&L Gates LLP", "Wenderoth, Lind & Ponack, L.L.P.", "Finnegan, Henderson, Farabow, Garrett & Dunner, LLP",
    "Conley Rose, P.C.", "Dinsmore & Shohl LLP", "BakerHostetler", "Rimon P.C.", "Lee & Hayes, P.C.", "GREENBLUM & BERNSTEIN, P.L.C.",
    "Harrity & Harrity, LLP", "Volpe Koenig", "Canon U.S.A., Inc. IP Division", "Lewis Roca Rothgerber Christie LLP",
    "Dickinson Wright PLLC", "Wolf, Greenfield & Sacks, P.C.", "NORTON ROSE FULBRIGHT US LLP", "Buchanan Ingersoll & Rooney PC",
    "McDonnell Boehnen Hulbert & Berghoff LLP", "Quarles & Brady LLP", "Maier & Maier, PLLC", "Lerner David LLP",
    "Barnes & Thornburg LLP", "Bryan Cave Leighton Paisner LLP", "Alston & Bird LLP", "Kowert, Hood, Munyon, Rankin & Goetzel, P.C.",
    "MARSHALL, GERSTEIN & BORUN LLP", "Maschoff Brennan", "ScienBiziP, P.C.", "Brooks Kushman P.C.", "The Webb Law Firm",
    "DLA Piper LLP (US)", "Armstrong Teasdale LLP", "Scully, Scott, Murphy & Presser, P.C.", "Nath, Goldberg & Meyer",
    "Shumaker & Sieffert, P.A.", "Faegre Drinker Biddle & Reath LLP", "F. CHAU & ASSOCIATES, LLC", "Bookoff McAndrews, PLLC",
    "CHIP LAW GROUP", "Amin, Turocy & Watson, LLP", "NSIP Law", "Fenwick & West LLP", "Dorsey & Whitney LLP",
    "Fletcher Yoder, P.C.", "Lowenstein Sandler LLP", "Honigman LLP", "Procopio, Cory, Hargreaves & Savitch LLP",
    "Jefferson IP Law, LLP", "Bayramoglu Law Offices LLC", "Haley Guiliano LLP", "McClure, Qualey & Rodack, LLP", "WHDA, LLP",
    "Sheridan Ross P.C.", "Young Basile Hanlon & MacFarlane, P.C.", "ANOVA LAW GROUP, PLLC", "Carter, DeLuca & Farrell LLP",
    "SNELL & WILMER L.L.P.", "Mintz Levin Cohn Ferris Glovsky and Popeo, P.C.", "Myers Bigel, P.A.", "Baker Botts L.L.P.",
    "Fox Rothschild LLP", "LEE, HONG, DEGERMAN, KANG & WAIMEY", "Sheppard Mullin Richter & Hampton LLP", "KDW Firm PLLC",
    "McCoy Russell LLP", "Keating & Bennett, LLP", "Klarquist Sparkman, LLP", "COOLEY LLP",
    "Meunier Carlin & Curfman LLC", "Shook, Hardy & Bacon L.L.P.", "Cooper Legal Group, LLC", "Pillsbury Winthrop Shaw Pittman LLP",
    "KILE PARK REED & HOUTTEMAN PLLC", "IP & T GROUP LLP", "Troutman Pepper Hamilton Sanders LLP", "Eversheds Sutherland (US) LLP",
    "Morrison & Foerster LLP", "NICHOLSON DE VOS WEBSTER & ELLIOTT LLP", "McAndrews, Held & Malloy, Ltd.",
    "Renner, Otto, Boisselle & Sklar, LLP", "Wilson Sonsini Goodrich & Rosati", "Lempia Summerfield Katz LLC",
    "Christensen O'Connor Johnson Kindness PLLC", "Greer, Burns & Crain, Ltd.", "Cozen O'Connor", "NIXON PEABODY LLP",
    "Duane Morris LLP", "McCarter & English, LLP", "Pearne & Gordon LLP", "Rankin, Hill & Clark LLP", "Locke Lord LLP",
    "Hanley, Flight & Zimmerman, LLC", "Kim & Stewart LLP", "Global IP Counselors, LLP", "FisherBroyles, LLP",
    "Price Heneveld LLP", "Fitch, Even, Tabin & Flannery LLP", "Schwabe, Williamson & Wyatt, P.C.", "Husch Blackwell LLP",
    "Potomac Law Group, PLLC", "Thomas | Horstemeyer, LLP", "Withrow & Terranova, PLLC", "METIS IP LLC",
    "QUALCOMM Incorporated", "Carlson, Gaskey & Olds, P.C.", "Eschweiler & Potashnik, LLC", "HSML P.C.", "PV IP PC",
    "Blank Rome LLP", "Alleman Hall & Tuttle LLP", "Weaver Austin Villeneuve & Sampson LLP", "Frost Brown Todd LLP",
    "Sage Patent Group", "Van Pelt, Yi & James LLP", "Murphy, Bilak & Homiller, PLLC", "Westman, Champlin & Koehler, P.A.",
    "IPUSA, PLLC", "Rothwell, Figg, Ernst & Manbeck, P.C.", "Muir Patent Law, PLLC", "MH2 TECHNOLOGY LAW GROUP LLP",
    "TraskBritt", "Edell, Shapiro & Finnan, LLC", "LOZA & LOZA, LLP", "Schmeiser, Olsen & Watts LLP", "Tucker Ellis LLP",
    "Tarolli, Sundheim, Covell & Tummino LLP", "SALIWANCHIK, LLOYD & EISENSCHENK", "Ladas & Parry, LLP",
    "Gray Ice Higdon", "MATTINGLY & MALUR, PC", "ROSSI, KIMMS & McDOWELL LLP", "CKC & Partners Co., LLC", "WPAT, P.C",
    "Weisberg I.P. Law, P.A.", "Syncoda LLC", "Seager, Tufte & Wickhem, LLP", "BAYES PLLC", "Norton Rose Fulbright Canada LLP",
    "Paratus Law Group, PLLC", "Taft Stettinius & Hollister LLP", "Guntin & Gust, PLC", "Shumaker, Loop & Kendrick, LLP",
    "NKL Law", "Lorenz & Kopf LLP", "KED & ASSOCIATES, LLP", "Li & Cai Intellectual Property (USA) Office",
    "Lippes Mathias LLP", "Ballard Spahr LLP", "The Marbury Law Group, PLLC", "Leason Ellis LLP", "Goodwin Procter LLP",
    "Maginot, Moore & Beck LLP", "Hodgson Russ LLP", "Posz Law Group, PLC", "Rabin & Berdo, P.C.", "Brake Hughes Bellermann LLP",
    "Ryan, Mason & Lewis, LLP", "Bozicevic, Field & Francis LLP", "W&G Law Group", "Calfee, Halter & Griswold LLP",
    "Brooks, Cameron & Huebsch, PLLC", "Panitch Schwarze Belisario & Nadel LLP", "Kunzler Bean & Adamson",
    "Moore & Van Allen PLLC", "SQUIRE PATTON BOGGS (US) LLP", "Botos Churchill IP Law LLP", "Mueting Raasch Group",
    "Caldwell Intellectual Property Law", "Innovation Counsel LLP", "Kinney & Lange, P.A.", "Casimir Jones, S.C.",
    "Quinn IP Law", "Howard & Howard Attorneys PLLC", "Andrus Intellectual Property Law, LLP", "The Roy Gross Law Firm, LLC",
    "Dentons US LLP", "STAAS & HALSEY LLP", "Heslin Rothenberg Farley & Mesiti P.C.", "NEO IP", "Finch & Maloney PLLC",
    "WILLIAM PARK & ASSOCIATES LTD.", "The Farrell Law Firm, P.C.", "PEARL COHEN ZEDEK LATZER BARATZ LLP",
    "Brownstein Hyatt Farber Schreck, LLP", "Treyz Law Group, P.C.", "Vorys, Sater, Seymour and Pease LLP", "Jones Day",
    "Neal, Gerber & Eisenberg LLP", "Suiter Swantz IP", "Sprinkle IP Law Group", "Foley Hoag LLP", "Vivacqua Crane, PLLC",
    "Garlick & Markison", "Keller Preece PLLC", "Arch & Lake LLP", "Warner Norcross + Judd LLP", "FIG. 1 Patents",
    "LUCAS & MERCANTI, LLP", "Vista IP Law Group, LLP", "Westbridge IP LLC", "Stinson LLP", "Reising Ethington P.C.",
    "Jordan IP Law, LLC", "WPAT LAW", "Davis Wright Tremaine LLP", "Holtz, Holtz & Volek PC", "Rutan & Tucker LLP",
    "Lathrop GPM LLP", "Riverside Law LLP", "Hunton Andrews Kurth LLP", "Collard & Roe, P.C.", "Jackson Walker L.L.P.",
    "Thompson Hine LLP", "Burris Law, PLLC", "Flaster Greenberg P.C.", "Getz Balich LLC", "Esplin & Associates, PC",
    "Shay Glenn LLP", "Hoffman Warnick LLC", "NovoTechIP International PLLC", "Hamilton, Brook, Smith & Reynolds, P.C.",
    "Bridgeway IP Law Group, PLLC", "Rosenberg, Klein & Lee", "Kirton McConkie", "HAYES SOLOWAY P.C.",
    "Calderon Safran & Wright P.C.", "McDermott Will & Emery LLP", "Crowe & Dunlevy LLC", "Millen, White, Zelano & Branigan P.C.",
    "Artegis Law Group, LLP", "Stites & Harbison, PLLC", "Bradley Arant Boult Cummings LLP", "Wood Herron & Evans LLP",
    "The Small Patent Law Group LLC", "HOUTTEMAN LAW LLC", "Condo Roccia Koptiw LLP", "Fay Kaplun & Marcin, LLP",
    "Burr & Forman LLP", "BURR PATENT LAW, PLLC", "Thorpe North & Western", "Paradice & Li LLP", "Neugeboren O'Dowd PC",
    "LRK PATENT LAW FIRM", "VLP Law Group LLP", "Reed Smith LLP", "Jaffery Watson Hamilton & DeSanctis LLP",
    "Zagorin Cave LLP", "Harter Secrest & Emery LLP", "Occhiuti & Rohlicek LLP", "Dechert LLP", "Miles & Stockbridge, P.C.",
    "Norris McLaughlin, P.A.", "Kaplan Breyer Schwarz, LLP", "Egbert, McDaniel & Swartz, PLLC", "Christensen, Fonder, Dardi & Herbert PLLC",
    "Stevens Law Group", "HSML P.C.", "Eckert Seamans Cherin & Mellott, LLC", "Maine Cernota & Curran", "Terrile, Cannatti & Chambers, LLP",
    "IPX PLLC", "BACON & THOMAS, PLLC", "Polygon IP, LLP", "Brannon Sowers & Cracraft PC", "Prol Intellectual Property Law, PLLC",
    "Sorell, Lenna & Schmidt, LLP", "Sandberg Phoenix & von Gontard P.C.", "Smith & Hopen, P.A.", "Dergosits & Noah LLP",
    "Clements Bernard Walker", "Kilyk & Bowersox, P.L.L.C.", "Porus IP LLC", "SLEMAN & LUND LLP", "Plager Schack LLP",
    "Smartpat PLC", "GTC Law Group PC & Affiliates", "Wissing Miller LLP", "East IP P.C.", "Barclay Damon LLP",
    "Intellectual Valley Law, P.C.", "Jenkins, Taylor & Hunt, P.A.", "Bell & Manning, LLC", "Haverstock & Owens, A Law Corporation",
    "Dierker & Kavanaugh, P.C.", "Henry M. Feiereisen LLC", "LANWAY IPR SERVICES", "MLO, a professional corp.",
    "N.V. Nederlandsch Octrooibureau", "INNOVATION CAPITAL LAW GROUP, LLP", "Spencer Fane LLP", "Beyer Law Group LLP",
    "Brown Rudnick LLP", "Umberg Zipser LLP", "Parker Justiss, P.C.", "Staniford Tomita LLP", "Farber LLC",
    "Duft & Bornsen, PC", "Idea Intellectual Limited", "FERENCE & ASSOCIATES LLC", "Smith Baluch LLP", "Culhane PLLC",
    "Reches Patents", "Stoel Rives LLP", "DUNLAP CODDING, P.C.", "Chen Yoshimura LLP", "McNees Wallace & Nurick LLC",
    "Weber Rosselli & Cannon LLP", "Edwards Neils LLC", "Fresh IP PLC", "NIELDS, LEMACK & FRAME, LLC",
    "Carstens, Allen & Gourley, LLP", "United One Law Group LLC", "Mendelsohn Dunleavy, P.C.", "Walter Ottesen, P.A.",
    "Nolte Lackenbach Siegel", "Bochner PLLC", "Fish IP Law, LLP", "Apex Attorneys at Law, LLP", "Weaver IP L.L.C.",
    "Mahamedi IP Law LLP", "Farjami & Farjami LLP", "Morse, Barnes-Brown & Pendleton, P.C.", "Buchalter",
    "Lee IP Law, P.C.", "MORI & WARD, LLP", "The Pattani Law Group", "True Shepherd LLC", "Insight Law Group, PLLC",
    "Anderson Gorecki LLP", "MaxGoLaw PLLC", "Ward Law Office LLC", "UB Greensfelder LLP", "Fleit Intellectual Property Law",
    "Gardner, Linn, Burkhart & Ondersma LLP", "Bracewell LLP", "CPST Intellectual Property Inc.", "McNeill PLLC",
    "Ade & Company Inc.", "Notaro, Michalos & Zaccaria P.C.", "Boudwin Intellectual Property Law, LLC",
    "Gesmer Updegrove LLP", "Dennemeyer & Associates LLC", "AlphaPatent Associates Ltd.", "Williams Mullen",
    "Ahmann Kloke LLP", "Ware, Fressola, Maguire & Barber LLP", "Miller Johnson", "Wood, Phillips, Katz, Clark & Mortimer",
    "Stanek Lemon Crouse & Meeks, PA", "Muirhead and Saturnelli, LLC", "GableGotwals", "Branch Partners PLLC",
    "Chalker Flores, LLP", "Park, Vaughan, Fleming & Dowler LLP", "Best & Flanagan LLP", "Joywin IP Law PLLC",
    "Whitmyer IP Group LLC", "KW Law, LLP", "Emerson, Thomson & Bennett, LLC", "Elmore Patent Law Group, P.C.",
    "Cognition IP, P.C.", "JMB Davis Ben-David", "Crawford Maunu PLLC", "ARC IP Law, PC", "Flagship Patents",
    "Laine IP Oy", "Sanchelima & Associates, P.A.", "Shackelford, McKinley & Norton, LLP", "RAPHAEL BELLUM PLLC",
    "Neustel Law Offices", "Perman & Green, LLP", "Miller Nash LLP", "Troutman Pepper Hamilton Sanders LLP (Rochester)",
    "Dykema Gossett PLLC", "2SPL Patent Attorneys PartG mbB", "Baker, Donelson, Bearman, Caldwell & Berkowitz PC",
    "Wang Law Firm, Inc.", "Bose McKinney & Evans LLP", "North Weber & Baugh", "Commvault Systems, Inc.",
    "Getech Law LLC", "Walters & Wasylyna LLC", "Burns & Levinson LLP", "HEA LAW PLLC", "Gilliam IP PLLC",
    "Law Office of Michael Chen", "PABST PATENT GROUP LLP", "Robinson + Cole LLP", "Birchwood IP",
    "TechLaw Ventures, PLLC", "DITTHAVONG, STEINER & MLOTKOWSKI", "Withers Bergman LLP", "USCH Law, PC",
    "Morris, Manning & Martin, LLP", "Patent 360", "CBM PATENT CONSULTING, LLC", "MATTHIAS SCHOLL P.C.",
    "Skaar Ulbrich Macari, P.A.", "Nokia Technologies Oy", "Otterstedt & Kammer PLLC", "Carr & Ferrell LLP",
    "Gates & Cooper LLP", "Grogan, Tuccillo & Vanderleeden LLP", "Benesch Friedlander Coplan & Aronoff LLP",
    "KIMBERLY-CLARK WORLDWIDE, INC.", "Ascenda Law Group, PC", "Wilmer Cutler Pickering Hale and Dorr LLP",
    "Caesar Rivise, PC", "Fujitsu Intellectual Property Center", "Baker & McKenzie LLP", "Chiesa Shahinian & Giantomasi PC",
    "Downs Rachlin Martin PLLC", "Howson & Howson LLP", "IP Business Solutions, LLC", "Chrisman Gallo Tochtrop LLC",
    "Pauley Erickson & Swanson", "McNeill Baur PLLC", "Han IP PLLC", "IPRTOP LLC", "Straub & Straub",
    "Johnson, Marcou, Isaacs & Nix, LLC", "CR MILES P.C.", "Bold IP PLLC", "Weide & Miller, Ltd.", "Ansari Katiraei LLP",
    "Patshegen IP", "Jacobson Holman PLLC", "SoCal IP Law Group LLP", "Thompson Coburn LLP", "Invention To Patent Services",
    "Dorton & Willis, LLP", "Hartman Global IP Law", "Goodhue, Coleman & Owens, P.C.", "Byrne Poh LLP", "WTA Patents",
    "Hoxie & Associates LLC", "John Rizvi, P.A.—The Patent Professor®", "Henry Patent Law Firm PLLC", "NOVAK DRUCE CARROLL LLP",
    "Novel IP", "FLYNN THIEL, P.C.", "KUSNER & JAFFE", "AP3 Law Firm PLLC", "McHale & Slavin, P.A.", "Verrill Dana, LLP",
    "Clark Hill PLC", "KONRAD, RAYNES, DAVDA & VICTOR LLP", "Thayne and Davis LLC", "Benoit & Cote Inc.", "Zhong Law, LLC",
    "RowanTree Law Group, PLLC", "Whitaker Chalk Swindle & Schwartz PLLC", "ASLAN LAW, P.C.", "Kramer Levin Naftalis & Frankel LLP",
    "Johnson & Johnson Surgical Vision, Inc.", "STIP Law Group, LLC", "Hoffberg & Associates", "Aka Chan LLP",
    "Law Office of Katsuhiro Arai", "Scale LLP", "Paschall & Associates, LLC", "Griggs Bergen LLP", "DICKINSON WRIGHT RLLP",
    "Cramer Patent & Design PLLC", "Imperium Patent Works", "Arrigo, Lee, Guttman & Mouta-Bellum LLP", "LaBatt, LLC",
    "IP Legal Services, LLC", "GrowIP Law Group LLC", "Strategic Patents, P.C.", "PatentPC/PowerPatent", "Inskeep IP Group, Inc.",
    "Orbit IP, LLP", "Taylor IP, P.C.", "DiBerardino McGovern IP Group LLC", "JAQUEZ LAND GREENHAUS & McFARLAND LLP",
    "Pramudji Law Group PLLC", "Yao Legal Services, Inc."
]

# Alias mapping for known law firm variations and mergers
alias_mapping = {
    "Arent Fox LLP": "ArentFox Schiff LLP",
    "Foley IP": "Foley & Lardner LLP",
    "Foley Hoag, LLP SeaPort West": "Foley Hoag LLP",
    "Foley Hoag, LLP Seaport West": "Foley Hoag LLP",
    "Baker & Hostetler LLP": "BakerHostetler",
    "Baker Hostetler LLP": "BakerHostetler",
    "Wilson Sonsini Good & Rosati PC": "Wilson Sonsini Goodrich & Rosati",
    "Wilson, Sonsini, Goodrich, and Rosati": "Wilson Sonsini Goodrich & Rosati",
    "Wilson Sonsini Goodrich & Rosati P.C.": "Wilson Sonsini Goodrich & Rosati",
    "Birch Steward Kolasch and Birch, LLP": "Birch, Stewart, Kolasch & Birch, LLP",
    "Muncy, Geissler, Olds & Lowe, PLLC": "Muncy, Geissler, Olds & Lowe, P.C.",
    "K & L Gates LLP": "K&L Gates LLP",
    "Kilpatrick Stocton & Townsend LLP": "Kilpatrick Townsend & Stockton LLP",
    "KILPATRICK TOWNSEND & STOCKTON, LLP": "Kilpatrick Townsend & Stockton LLP",
    "Kilpatrick Townsend & Stockton LLP (retired)": "Kilpatrick Townsend & Stockton LLP",
    "Knobbe Martens Olson & Bear LLP": "Knobbe, Martens, Olson & Bear, LLP",
    "Knobbe Martens Olson and Bear, LLP": "Knobbe, Martens, Olson & Bear, LLP",
    "Canon USA Inc.": "Canon U.S.A., Inc. IP Division",
    "Culhane Meadows PLLC": "Culhane PLLC",
    "Fish & Richardson, P.C.": "Fish & Richardson P.C.",
    "Fish & Richardon, P.C.": "Fish & Richardson P.C.",
    "Fish and Richardson P.C.": "Fish & Richardson P.C.",
    "Levine Bagade Han LLP": "Han IP PLLC",
    "Han Santos, PLLC": "Han IP PLLC",
    "Finnegan, Henderson, Farabow, Garret & Dunner": "Finnegan, Henderson, Farabow, Garrett & Dunner, LLP",
    "Mintz Levin, PC": "Mintz Levin Cohn Ferris Glovsky and Popeo, P.C.",
    "Mintz, Levin, Cohn, Ferris, Glovsky and Popeo, P.C.": "Mintz Levin Cohn Ferris Glovsky and Popeo, P.C.",
    "MRG IP LAW, P.A. DBA Mueting Raasch Group": "Mueting Raasch Group",
    "Womble Bond Dickinson LLP": "Womble Bond Dickinson (US) LLP",
    "Womble, Bond, and Dickinson": "Womble Bond Dickinson (US) LLP",
    "Slater & Matsil, L.L.P.": "Slater Matsil, LLP",
    "Quarels & Brady LLP": "Quarles & Brady LLP",
    "Alston & Bird One Atlantic Center": "Alston & Bird LLP",
    "Calfee, Halter & Griswold LLP The Calfee Building": "Calfee, Halter & Griswold LLP",
    "Bryan Cave, LLP": "Bryan Cave Leighton Paisner LLP",
    "Dorsey & Whitney, LLP": "Dorsey & Whitney LLP",
    "Procopio, Cory, Hargreaves & Savitch": "Procopio, Cory, Hargreaves & Savitch LLP",
    "Novo TechIP International PLLC": "NovoTechIP International PLLC",
    "Maier and Maier": "Maier & Maier, PLLC",
    "Pillsbury Winthrop Shaw Pittman": "Pillsbury Winthrop Shaw Pittman LLP",  # Added
    "Rutan & Tucker, LLP": "Rutan & Tucker, LLP"  # Added
}

# Log original column names and row count
logging.info(f"Columns in all_data: {all_data.columns.tolist()}")
logging.info(f"Total rows in all_data: {len(all_data)}")
logging.info(f"Loaded {len(correct_names)} correct organization names")

# Validate required columns
required_columns = [org_col, reg_col]
missing_columns = [col for col in required_columns if col not in all_data.columns]
if missing_columns:
    logging.error(f"Missing columns in all_data.xlsx: {missing_columns}")
    exit(1)

# Log rows with NaN or empty values
nan_org = all_data[all_data[org_col].isna()]
empty_org = all_data[all_data[org_col].str.strip() == '']
nan_reg = all_data[all_data[reg_col].isna()]
logging.info(f"Rows with NaN in {org_col}: {len(nan_org)}")
logging.info(f"Rows with empty string in {org_col}: {len(empty_org)}")
logging.info(f"Rows with NaN in {reg_col}: {len(nan_reg)}")
if len(nan_org) > 0 or len(empty_org) > 0 or len(nan_reg) > 0:
    nan_rows = all_data[all_data[org_col].isna() | (all_data[org_col].str.strip() == '') | all_data[reg_col].isna()].copy()
    nan_rows.loc[nan_rows[org_col].isna(), 'drop_reason'] = 'NaN org_name'
    nan_rows.loc[nan_rows[org_col].str.strip() == '', 'drop_reason'] = 'Empty org_name'
    nan_rows.loc[nan_rows[reg_col].isna(), 'drop_reason'] = 'NaN reg_code'
    nan_rows.to_csv("nan_rows.csv", index=False)
    logging.info(f"Saved {len(nan_rows)} rows with NaN/empty to nan_rows.csv")

# Clean data: Drop rows with missing or empty org_col and reg_col
all_data = all_data.dropna(subset=[org_col, reg_col])
all_data = all_data[all_data[org_col].str.strip() != '']
logging.info(f"After cleaning, {len(all_data)} rows remain")

# Check for duplicate reg_code values in input data
duplicate_reg_codes = all_data[all_data[reg_col].duplicated(keep=False)]
if len(duplicate_reg_codes) > 0:
    logging.warning(f"Found {len(duplicate_reg_codes)} rows with duplicate Reg Code values")
    duplicate_reg_codes.to_csv("duplicate_reg_codes.csv", index=False)
    logging.info(f"Saved duplicate Reg Code rows to duplicate_reg_codes.csv")

# Preprocess function for better matching
def preprocess_name(name):
    if not isinstance(name, str) or not name.strip():
        return ""
    name = name.lower().strip()
    name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
    generics = r'\b(llp|llc|pllc|pc|corp|company|firm|group|law|office|of|the|and|ip|intellectual|property|strategies|incorporated|inc|pa|p.c.|pllc|llp|llc|corp|company|usa|division|technologies|associates|attorneys|legal|services|university|center|school|texas|md|anderson|cancer|systems|northrop|grumman|communication|optical|corning|alley|valley|west|east|north|south|seaport|campus|drive|street|avenue|suite|floor|building|rllp|wrights|patent|innovative|research|department|bio|pharma|science|biomedical|technology|international|consulting)\b'
    name = re.sub(generics, '', name, flags=re.IGNORECASE)
    name = ' '.join(name.split())  # Normalize spaces
    return name

preprocessed_correct = {name: preprocess_name(name) for name in correct_names}

# ==== 3. Connect to MongoDB ====
try:
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    correct_collection = db['correct_orgs']
    logging.info(f"Connected to MongoDB: {db_name}, collection: {collection_name}")
except Exception as e:
    logging.error(f"Failed to connect to MongoDB: {e}")
    exit(1)

# Clear collection if specified
if clear_collection:
    confirm = input("Are you sure you want to clear the standardized_orgs collection? (yes/no): ")
    if confirm.lower() == "yes":
        collection.delete_many({})
        logging.info("Cleared standardized_orgs collection")
    else:
        logging.info("Skipped clearing standardized_orgs collection")

# Insert correct names if not already
if correct_collection.count_documents({}) == 0:
    correct_docs = [{"correct_name": name} for name in correct_names]
    correct_collection.insert_many(correct_docs)
    logging.info(f"Inserted {len(correct_names)} correct names into 'correct_orgs'")

# Helper to get best match with stricter criteria
def get_best_match(org_name):
    # Check if org_name exactly matches a correct_name
    if org_name in correct_names:
        return org_name, 100

    # Check if org_name is in alias_mapping and maps to a law firm in correct_names
    if org_name in alias_mapping and alias_mapping[org_name] in correct_names:
        return alias_mapping[org_name], 100

    # If not a law firm (not in correct_names or alias_mapping), retain original name
    if org_name not in correct_names and org_name not in alias_mapping:
        preprocessed_org = preprocess_name(org_name)
        if not preprocessed_org:
            return org_name, 0
        match, score, _ = process.extractOne(
            preprocessed_org, list(preprocessed_correct.values()), scorer=fuzz.token_set_ratio
        )
        # Stricter threshold
        if score <= 90:
            return org_name, 0
        # Map back to original correct name
        original_match = next((key for key, value in preprocessed_correct.items() if value == match), None)
        
        # Stricter core word check: Require at least two common non-generic words
        org_words = set(preprocessed_org.split())
        match_words = set(preprocess_name(original_match).split())
        common_core = org_words & match_words
        if len(common_core) < 2:
            return org_name, 0  # Less than two common core words, retain original
        
        # Log potential matches for review
        if score > 80:
            logging.info(f"Potential fuzzy match: {org_name} -> {original_match} (score: {score})")
        
        return original_match, score

    return org_name, 0

# ==== 5. Iterate and prepare data using iterrows() ====
batch_size = 1000
records_to_insert = []
error_rows = []
duplicate_skipped = []
mismatched_rows = []
processed_count = 0
skipped_count = 0

for i, row in all_data.iterrows():
    try:
        org_name = row[org_col]
        reg_code = row[reg_col]

        # Log values for debugging first few rows
        if processed_count + skipped_count < 5:
            logging.info(f"Row {i}: org_name='{org_name}' (type: {type(org_name)}), reg_code={reg_code} (type: {type(reg_code)})")

        # Skip if reg_code is invalid
        if pd.isna(reg_code):
            logging.warning(f"Skipping row {i}: Invalid reg_code - Value: {reg_code}")
            error_rows.append({"row_index": i, "data": row.to_dict(), "error": f"Invalid reg_code: {reg_code}"})
            skipped_count += 1
            continue
        reg_code_str = str(int(reg_code)) if isinstance(reg_code, (int, float)) and reg_code == int(reg_code) else str(reg_code)

        # Skip if reg_code exists
        if collection.find_one({"regCode": reg_code_str}):
            logging.debug(f"Skipping duplicate regCode: {reg_code_str}")
            duplicate_skipped.append({"row_index": i, "data": row.to_dict(), "error": f"Duplicate regCode: {reg_code_str}"})
            skipped_count += 1
            continue

        best_match, score = get_best_match(org_name)
        
        doc = {
            "regCode": reg_code_str,
            "name": row.get("Name"),
            "organization_original": org_name,
            "standardized_org": best_match,
            "similarity_score": score,
            "addressLine1": row.get("Address Line 1"),
            "addressLine2": row.get("Address Line 2"),
            "city": row.get("City"),
            "state": row.get("State"),
            "country": row.get("Country"),
            "zipcode": row.get("Zipcode"),
            "phoneNumber": row.get("Phone Number"),
            "agent_or_attorney": row.get("Agent/Attorney"),
            "processedAt": datetime.now(timezone.utc).isoformat()
        }

        # Log mismatched rows for review
        if score > 0 and best_match != org_name:
            mismatched_rows.append({
                "row_index": i,
                "organization_original": org_name,
                "standardized_org": best_match,
                "similarity_score": score
            })

        records_to_insert.append(doc)
        processed_count += 1
        
        if len(records_to_insert) >= batch_size:
            try:
                collection.insert_many(records_to_insert)
                logging.info(f"Inserted batch of {len(records_to_insert)} records")
                records_to_insert = []
            except Exception as e:
                logging.error(f"Error inserting batch at row {i}: {e}")
                error_rows.append({"row_index": i, "data": row.to_dict(), "error": f"Batch insert failed: {e}"})
                skipped_count += 1
    except Exception as e:
        logging.error(f"Error processing row {i}: {e} - Row data: {row.to_dict()}")
        error_rows.append({"row_index": i, "data": row.to_dict(), "error": str(e)})
        skipped_count += 1
        continue

# Insert remaining records
if records_to_insert:
    try:
        collection.insert_many(records_to_insert)
        logging.info(f"Inserted final batch of {len(records_to_insert)} records")
    except Exception as e:
        logging.error(f"Error inserting final batch: {e}")

logging.info(f"Total processed rows: {processed_count}, skipped: {skipped_count}")

# Save error rows, duplicates, and mismatched rows
try:
    if error_rows:
        error_df = pd.DataFrame(error_rows)
        error_df.to_csv("error_rows.csv", index=False)
        error_df.to_excel("error_rows.xlsx", index=False)
        logging.info(f"Saved {len(error_rows)} problematic rows to error_rows.csv and error_rows.xlsx")
    if duplicate_skipped:
        dup_df = pd.DataFrame(duplicate_skipped)
        dup_df.to_csv("duplicate_skipped.csv", index=False)
        dup_df.to_excel("duplicate_skipped.xlsx", index=False)
        logging.info(f"Saved {len(duplicate_skipped)} duplicate regCode rows to duplicate_skipped.csv and duplicate_skipped.xlsx")
    if mismatched_rows:
        mismatch_df = pd.DataFrame(mismatched_rows)
        mismatch_df.to_csv("mismatched_rows.csv", index=False)
        mismatch_df.to_excel("mismatched_rows.xlsx", index=False)
        logging.info(f"Saved {len(mismatched_rows)} mismatched rows to mismatched_rows.csv and mismatched_rows.xlsx")
except Exception as e:
    logging.error(f"Error saving error_rows, duplicates, or mismatched_rows: {e}")

# Export standardized_orgs to Excel
try:
    cursor = collection.find({})
    df = pd.DataFrame(list(cursor))
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])
    df.to_excel("standardized_orgs.xlsx", index=False)
    logging.info("Exported standardized_orgs to standardized_orgs.xlsx")
except Exception as e:
    logging.error(f"Error exporting to Excel: {e}")

# Log total records inserted
try:
    total_inserted = collection.count_documents({})
    logging.info(f"Total records in standardized_orgs: {total_inserted}")
except Exception as e:
    logging.error(f"Error counting documents: {e}")

# ==== 6. Close connection ====
client.close()
logging.info("MongoDB connection closed")