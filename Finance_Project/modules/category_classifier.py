"""
Smart Indian UPI Transaction Category Classifier
Handles real-world UPI payee strings, merchant names, VPAs, and narrations.
"""

import re
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# MERCHANT LOOKUP — High-precision exact/prefix matches
# Each entry: (regex_pattern, category, subcategory, display_name)
# ══════════════════════════════════════════════════════════════════════════════

MERCHANT_DB = [
    # ── Food Delivery ────────────────────────────────────────────────────────
    (r'zomato',                      'Food & Dining',          'Food Delivery',   'Zomato'),
    (r'swiggy',                      'Food & Dining',          'Food Delivery',   'Swiggy'),
    (r'magicpin',                    'Food & Dining',          'Food Delivery',   'Magicpin'),
    (r'eatsure',                     'Food & Dining',          'Food Delivery',   'EatSure'),
    (r'dunzo',                       'Food & Dining',          'Food Delivery',   'Dunzo'),
    (r'thrive',                      'Food & Dining',          'Food Delivery',   'Thrive'),
    (r'shadowfax',                   'Food & Dining',          'Food Delivery',   'Shadowfax'),
    (r'freshmenu',                   'Food & Dining',          'Food Delivery',   'FreshMenu'),
    (r'box8',                        'Food & Dining',          'Food Delivery',   'Box8'),
    (r'oven\s*story|ovenstory',      'Food & Dining',          'Food Delivery',   'Oven Story'),
    (r'faasos',                      'Food & Dining',          'Food Delivery',   'Faasos'),
    (r'behrouz',                     'Food & Dining',          'Food Delivery',   'Behrouz Biryani'),
    (r'pizza\s*hut|pizzahut',        'Food & Dining',          'Restaurants',     'Pizza Hut'),
    (r'dominos?|domino\'?s',         'Food & Dining',          'Restaurants',     "Domino's"),
    (r'mcdonalds?|mcd\b|mcdonald',   'Food & Dining',          'Restaurants',     "McDonald's"),
    (r'burger\s*king|burgerking',    'Food & Dining',          'Restaurants',     'Burger King'),
    (r'kfc\b',                       'Food & Dining',          'Restaurants',     'KFC'),
    (r'subway\b',                    'Food & Dining',          'Restaurants',     'Subway'),
    (r'starbucks',                   'Food & Dining',          'Café',            'Starbucks'),
    (r'cafe\s*coffee\s*day|ccd\b',   'Food & Dining',          'Café',            'Café Coffee Day'),
    (r'chai\s*point|chaipoint',      'Food & Dining',          'Café',            'Chai Point'),
    (r'third\s*wave\s*coffee',       'Food & Dining',          'Café',            'Third Wave Coffee'),
    (r'bira\b|bira91',               'Food & Dining',          'Beverages',       'Bira 91'),
    (r'haldirams?',                  'Food & Dining',          'Restaurants',     "Haldiram's"),
    (r'barbeque\s*nation|bbq\s*ntn', 'Food & Dining',          'Restaurants',     'Barbeque Nation'),

    # ── Groceries & Supermarkets ──────────────────────────────────────────────
    (r'bigbasket|big\s*basket',      'Groceries',              'Online Grocery',  'BigBasket'),
    (r'blinkit|grofers',             'Groceries',              'Quick Commerce',  'Blinkit'),
    (r'zepto\b',                     'Groceries',              'Quick Commerce',  'Zepto'),
    (r'instamart|swiggy.*insta',     'Groceries',              'Quick Commerce',  'Instamart'),
    (r'jiomart|jio\s*mart',          'Groceries',              'Online Grocery',  'JioMart'),
    (r'nature[s\']*\s*basket',       'Groceries',              'Premium Grocery', "Nature's Basket"),
    (r'dmart\b|d[\s-]?mart',         'Groceries',              'Supermarket',     'D-Mart'),
    (r'reliance\s*(fresh|smart|retail)', 'Groceries',          'Supermarket',     'Reliance Retail'),
    (r'more\s*retail|more\.co',      'Groceries',              'Supermarket',     'More Retail'),
    (r'spencers',                    'Groceries',              'Supermarket',     "Spencer's"),
    (r'star\s*(bazaar|market)',       'Groceries',              'Supermarket',     'Star Bazaar'),
    (r'licious\b',                   'Groceries',              'Meat & Seafood',  'Licious'),
    (r'fresho|isha\s*farms',         'Groceries',              'Fresh Produce',   'FreshO'),
    (r'country\s*delight',           'Groceries',              'Dairy',           'Country Delight'),
    (r'milkbasket',                  'Groceries',              'Dairy',           'Milkbasket'),

    # ── Shopping / E-Commerce ─────────────────────────────────────────────────
    (r'amazon(?!.*pay)',             'Shopping',               'E-Commerce',      'Amazon'),
    (r'flipkart',                    'Shopping',               'E-Commerce',      'Flipkart'),
    (r'myntra',                      'Shopping',               'Fashion',         'Myntra'),
    (r'ajio\b',                      'Shopping',               'Fashion',         'AJIO'),
    (r'nykaa',                       'Shopping',               'Beauty & Personal','Nykaa'),
    (r'meesho',                      'Shopping',               'E-Commerce',      'Meesho'),
    (r'snapdeal',                    'Shopping',               'E-Commerce',      'Snapdeal'),
    (r'shopsy',                      'Shopping',               'E-Commerce',      'Shopsy'),
    (r'tata\s*cliq|tatacliq',        'Shopping',               'E-Commerce',      'Tata CLiQ'),
    (r'firstcry',                    'Shopping',               'Baby & Kids',     'FirstCry'),
    (r'purplle',                     'Shopping',               'Beauty & Personal','Purplle'),
    (r'mamaearth',                   'Shopping',               'Beauty & Personal','Mamaearth'),
    (r'plum\b',                      'Shopping',               'Beauty & Personal','Plum'),
    (r'bewakoof',                    'Shopping',               'Fashion',         'Bewakoof'),
    (r'clovia',                      'Shopping',               'Fashion',         'Clovia'),
    (r'boat\b|boataudio',            'Shopping',               'Electronics',     'boAt Audio'),
    (r'noise\b',                     'Shopping',               'Electronics',     'Noise'),
    (r'croma\b',                     'Shopping',               'Electronics',     'Croma'),
    (r'vijay\s*sales',               'Shopping',               'Electronics',     'Vijay Sales'),
    (r'reliance\s*digital',          'Shopping',               'Electronics',     'Reliance Digital'),
    (r'inr\s*deals|dealshare',       'Shopping',               'E-Commerce',      'DealShare'),

    # ── Travel & Transport ────────────────────────────────────────────────────
    (r'uber\b',                      'Travel & Transport',     'Cab',             'Uber'),
    (r'ola\s*(cabs?|electric)?',     'Travel & Transport',     'Cab',             'Ola'),
    (r'rapido\b',                    'Travel & Transport',     'Cab/Bike',        'Rapido'),
    (r'namma\s*yatri|yatri\b',       'Travel & Transport',     'Auto/Cab',        'Namma Yatri'),
    (r'indigo|interglobe\s*avia',    'Travel & Transport',     'Flight',          'IndiGo'),
    (r'air\s*india|airindia',        'Travel & Transport',     'Flight',          'Air India'),
    (r'spicejet',                    'Travel & Transport',     'Flight',          'SpiceJet'),
    (r'vistara',                     'Travel & Transport',     'Flight',          'Vistara'),
    (r'akasa\s*air',                 'Travel & Transport',     'Flight',          'Akasa Air'),
    (r'irctc|indian\s*railway',      'Travel & Transport',     'Train',           'IRCTC'),
    (r'redbus',                      'Travel & Transport',     'Bus',             'RedBus'),
    (r'abhibus',                     'Travel & Transport',     'Bus',             'AbhiBus'),
    (r'ixigo',                       'Travel & Transport',     'Travel Booking',  'ixigo'),
    (r'makemytrip|mmt\b',            'Travel & Transport',     'Travel Booking',  'MakeMyTrip'),
    (r'goibibo',                     'Travel & Transport',     'Travel Booking',  'Goibibo'),
    (r'yatra\b',                     'Travel & Transport',     'Travel Booking',  'Yatra'),
    (r'cleartrip',                   'Travel & Transport',     'Travel Booking',  'Cleartrip'),
    (r'easemytrip',                  'Travel & Transport',     'Travel Booking',  'EaseMyTrip'),
    (r'oyo\s*(rooms?)?',             'Travel & Transport',     'Hotel',           'OYO'),
    (r'fabhotels|fab\s*hotels',      'Travel & Transport',     'Hotel',           'FabHotels'),
    (r'stayzi|treebo',               'Travel & Transport',     'Hotel',           'Treebo'),
    (r'airbnb',                      'Travel & Transport',     'Hotel',           'Airbnb'),
    (r'bmtc|best\s*bus|mthl|nmmt',   'Travel & Transport',     'Public Transport','City Bus'),
    (r'metro\s*rail|dmrc|bmrcl|nmrc|cmrl', 'Travel & Transport','Metro',         'Metro Rail'),
    (r'fastag|netc\b|ihmcl',         'Travel & Transport',     'Toll/FASTag',     'FASTag'),
    (r'petrol|fuel|hpcl|bpcl|iocl|essar\s*oil|shell|hp\s*pump', 
                                     'Travel & Transport',     'Fuel',            'Fuel/Petrol'),
    (r'nift\s*parking|park\+',       'Travel & Transport',     'Parking',         'Parking'),

    # ── Utilities & Bills ─────────────────────────────────────────────────────
    (r'mahadiscom|mseb\b|msedcl',    'Utilities & Bills',      'Electricity',     'MSEDCL'),
    (r'bescom\b',                    'Utilities & Bills',      'Electricity',     'BESCOM'),
    (r'tpddl|bses|adani\s*electric', 'Utilities & Bills',      'Electricity',     'Electricity'),
    (r'torrent\s*power',             'Utilities & Bills',      'Electricity',     'Torrent Power'),
    (r'piped\s*gas|mahanagar\s*gas|mgl\b|adani\s*gas|igl\b|gail', 
                                     'Utilities & Bills',      'Gas',             'Piped Gas'),
    (r'jio\b(?!mart)',               'Utilities & Bills',      'Mobile Recharge', 'Jio'),
    (r'airtel(?!\s*payment)',        'Utilities & Bills',      'Mobile/Broadband','Airtel'),
    (r'vi\b|vodafone|idea\b',        'Utilities & Bills',      'Mobile Recharge', 'Vi (Vodafone Idea)'),
    (r'bsnl\b',                      'Utilities & Bills',      'Mobile Recharge', 'BSNL'),
    (r'act\s*fibernet|act\b',        'Utilities & Bills',      'Broadband',       'ACT Fibernet'),
    (r'hathway|tikona',              'Utilities & Bills',      'Broadband',       'Broadband'),
    (r'tata\s*(sky|play)',           'Utilities & Bills',      'DTH',             'Tata Play'),
    (r'dish\s*tv|dishtv',            'Utilities & Bills',      'DTH',             'Dish TV'),
    (r'sun\s*direct|sundirect',      'Utilities & Bills',      'DTH',             'Sun Direct'),
    (r'd2h\b|videocon\s*d2h',        'Utilities & Bills',      'DTH',             'D2H'),
    (r'bmc\b|bbmp\b|municipal.*tax|property\s*tax', 
                                     'Utilities & Bills',      'Municipal Tax',   'Property Tax'),
    (r'water\s*(board|supply|tax)',  'Utilities & Bills',      'Water',           'Water Bill'),

    # ── Financial Services ────────────────────────────────────────────────────
    (r'sip\b|mutual\s*fund|mf\b',    'Investments',            'Mutual Fund SIP', 'Mutual Fund SIP'),
    (r'groww\b',                     'Investments',            'Stockbroker',     'Groww'),
    (r'zerodha|kite\b',              'Investments',            'Stockbroker',     'Zerodha'),
    (r'upstox',                      'Investments',            'Stockbroker',     'Upstox'),
    (r'angel\s*(one|broking)',       'Investments',            'Stockbroker',     'Angel One'),
    (r'5paisa',                      'Investments',            'Stockbroker',     '5paisa'),
    (r'kuvera|fisdom',               'Investments',            'Mutual Fund',     'Mutual Fund App'),
    (r'coin\.zerodha',               'Investments',            'Mutual Fund',     'Zerodha Coin'),
    (r'ppf\b|public\s*provident',    'Investments',            'PPF',             'PPF'),
    (r'nps\b|national\s*pension',    'Investments',            'NPS',             'NPS'),
    (r'gold\s*bond|sgb\b',           'Investments',            'Gold Bond',       'Sovereign Gold Bond'),
    (r'fd\b|fixed\s*deposit',        'Investments',            'Fixed Deposit',   'Fixed Deposit'),
    (r'rd\b|recurring\s*deposit',    'Investments',            'Recurring Dep.',  'Recurring Deposit'),
    (r'lic\b|life\s*insur',          'Insurance',              'Life Insurance',  'LIC'),
    (r'hdfc\s*life|sbi\s*life|max\s*life|kotak\s*life|tata\s*aia', 
                                     'Insurance',              'Life Insurance',  'Life Insurance'),
    (r'star\s*health|niva\s*bupa|care\s*(health|insur)|bajaj\s*allianz', 
                                     'Insurance',              'Health Insurance','Health Insurance'),
    (r'acko\b|go\s*digit|digit\s*insur', 
                                     'Insurance',              'Vehicle Insurance','Vehicle Insurance'),
    (r'loan\s*(emi|repay)|emi\b',    'Loan/EMI',               'Loan EMI',        'Loan EMI'),
    (r'credit\s*card.*pay|cc\s*pay', 'Loan/EMI',               'Credit Card Bill','Credit Card Bill'),

    # ── Healthcare ────────────────────────────────────────────────────────────
    (r'apollo\s*(pharmacy|health|hospital)?', 
                                     'Healthcare',             'Pharmacy/Hospital','Apollo'),
    (r'medplus',                     'Healthcare',             'Pharmacy',        'MedPlus'),
    (r'netmeds',                     'Healthcare',             'Online Pharmacy', 'Netmeds'),
    (r'1mg\b|tata\s*1mg',            'Healthcare',             'Online Pharmacy', '1mg'),
    (r'pharmeasy',                   'Healthcare',             'Online Pharmacy', 'PharmEasy'),
    (r'practo\b',                    'Healthcare',             'Doctor Consult',  'Practo'),
    (r'healthians',                  'Healthcare',             'Lab Tests',       'Healthians'),
    (r'thyrocare',                   'Healthcare',             'Lab Tests',       'Thyrocare'),
    (r'lal\s*path\s*labs?|lalpathlab', 
                                     'Healthcare',             'Lab Tests',       'Lal PathLabs'),

    # ── Education ─────────────────────────────────────────────────────────────
    (r'byju',                        'Education',              'EdTech',          "BYJU'S"),
    (r'unacademy',                   'Education',              'EdTech',          'Unacademy'),
    (r'vedantu',                     'Education',              'EdTech',          'Vedantu'),
    (r'upgrad|up\s*grad',            'Education',              'EdTech',          'upGrad'),
    (r'coursera',                    'Education',              'EdTech',          'Coursera'),
    (r'udemy',                       'Education',              'EdTech',          'Udemy'),
    (r'simplilearn',                 'Education',              'EdTech',          'Simplilearn'),
    (r'school\s*(fee|tuition)',       'Education',              'School Fee',      'School Fee'),
    (r'college\s*fee',               'Education',              'College Fee',     'College Fee'),

    # ── Entertainment & Subscriptions ─────────────────────────────────────────
    (r'netflix',                     'Entertainment',          'OTT',             'Netflix'),
    (r'amazon\s*prime|prime\s*video','Entertainment',          'OTT',             'Amazon Prime'),
    (r'hotstar|disney\s*\+|jio\s*cinema', 
                                     'Entertainment',          'OTT',             'Disney+/Hotstar'),
    (r'sony\s*(liv|liv)',            'Entertainment',          'OTT',             'SonyLIV'),
    (r'zee5\b|zee\s*5',              'Entertainment',          'OTT',             'ZEE5'),
    (r'mxplayer|mx\s*player',        'Entertainment',          'OTT',             'MX Player'),
    (r'spotify',                     'Entertainment',          'Music',           'Spotify'),
    (r'gaana\b|saavn\b|jiosaavn',    'Entertainment',          'Music',           'JioSaavn'),
    (r'wynk\b',                      'Entertainment',          'Music',           'Wynk Music'),
    (r'bookmyshow|bms\b',            'Entertainment',          'Movies/Events',   'BookMyShow'),
    (r'paytm\s*movie|ticketnew',     'Entertainment',          'Movies/Events',   'Movie Tickets'),
    (r'inox\b|pvr\b',                'Entertainment',          'Movies/Events',   'Cinema'),
    (r'steam\b',                     'Entertainment',          'Gaming',          'Steam'),
    (r'playstation|ps\s*store',      'Entertainment',          'Gaming',          'PlayStation'),
    (r'google\s*play',               'Entertainment',          'App Store',       'Google Play'),
    (r'apple\b',                     'Entertainment',          'App Store',       'Apple'),

    # ── Payment Wallets / Gateways ────────────────────────────────────────────
    (r'paytm(?!\s*(movie|mall|qr|merchant))', 
                                     'UPI Transfer',           'Wallet',          'Paytm'),
    (r'phonepe',                     'UPI Transfer',           'UPI/Wallet',      'PhonePe'),
    (r'gpay|google\s*pay',           'UPI Transfer',           'UPI',             'Google Pay'),
    (r'amazon\s*pay',                'UPI Transfer',           'Wallet',          'Amazon Pay'),
    (r'mobikwik',                    'UPI Transfer',           'Wallet',          'MobiKwik'),
    (r'airtel\s*payment|airtelpaymentsbank', 
                                     'UPI Transfer',           'Wallet',          'Airtel Payments'),
    (r'freecharge',                  'UPI Transfer',           'Wallet',          'FreeCharge'),

    # ── Banks / ATM ───────────────────────────────────────────────────────────
    (r'atm\s*wtdl|atm\s*with|cash\s*withdrawal', 
                                     'Cash & ATM',             'ATM Withdrawal',  'ATM Withdrawal'),
    (r'neft\b|rtgs\b|imps\b',        'Bank Transfer',          'Bank Transfer',   'Bank Transfer'),
    (r'hdfc\s*bank',                 'Bank Transfer',          'HDFC Bank',       'HDFC Bank'),
    (r'icici\s*bank',                'Bank Transfer',          'ICICI Bank',      'ICICI Bank'),
    (r'sbi\b|state\s*bank',          'Bank Transfer',          'SBI',             'SBI'),
    (r'axis\s*bank',                 'Bank Transfer',          'Axis Bank',       'Axis Bank'),
    (r'kotak\s*(mahindra)?\s*bank',  'Bank Transfer',          'Kotak Bank',      'Kotak Bank'),
    (r'yes\s*bank',                  'Bank Transfer',          'Yes Bank',        'Yes Bank'),
    (r'idfc\b|bandhan\b|rbl\b',      'Bank Transfer',          'Bank',            'Bank Transfer'),
    (r'indusind',                    'Bank Transfer',          'IndusInd Bank',   'IndusInd Bank'),
    (r'bob\b|bank\s*of\s*baroda',    'Bank Transfer',          'BoB',             'Bank of Baroda'),
    (r'pnb\b|punjab\s*national',     'Bank Transfer',          'PNB',             'Punjab National Bank'),
    (r'canara\s*bank',               'Bank Transfer',          'Canara Bank',     'Canara Bank'),
    (r'union\s*bank',                'Bank Transfer',          'Union Bank',      'Union Bank of India'),
    (r'federal\s*bank',              'Bank Transfer',          'Federal Bank',    'Federal Bank'),

    # ── Government / Tax ─────────────────────────────────────────────────────
    (r'income\s*tax|itr\b',          'Taxes & Government',     'Income Tax',      'Income Tax'),
    (r'gst\b',                       'Taxes & Government',     'GST',             'GST'),
    (r'epfo\b|pf\b|provident\s*fund','Investments',            'EPF/PF',          'Provident Fund'),
    (r'challan\b',                   'Taxes & Government',     'Government Fee',  'Government Challan'),
    (r'passport\b',                  'Taxes & Government',     'Passport',        'Passport Fee'),
    (r'aadhaar\b',                   'Taxes & Government',     'Aadhaar',         'Aadhaar Service'),

    # ── Salary / Income ───────────────────────────────────────────────────────
    (r'salary\b|sal\b',              'Income',                 'Salary',          'Salary Credit'),
    (r'payroll',                     'Income',                 'Salary',          'Payroll'),
    (r'interest\s*(credit|earned|paid|on dep)', 
                                     'Income',                 'Interest',        'Interest Earned'),
    (r'dividend\b',                  'Income',                 'Dividend',        'Dividend'),
    (r'cashback\b|cash\s*back',      'Income',                 'Cashback',        'Cashback'),
    (r'refund\b|reversal\b',         'Income',                 'Refund',          'Refund'),

    # ── Donations & Charity ───────────────────────────────────────────────────
    (r'donate|donation|charity|ngo\b|giveindia', 
                                     'Charity & Donations',    'Donation',        'Donation'),

    # ── Rent & Housing ────────────────────────────────────────────────────────
    (r'rent\b|rental\b',             'Housing & Rent',         'Rent',            'Rent Payment'),
    (r'maintenance\s*(fee|charge)',  'Housing & Rent',         'Maintenance',     'Society Maintenance'),
    (r'nobroker|nestaway|stanza',    'Housing & Rent',         'Rental Platform', 'Rental Service'),
]


# ══════════════════════════════════════════════════════════════════════════════
# COMPILED PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
_COMPILED = [
    (re.compile(pat, re.IGNORECASE), cat, subcat, display)
    for pat, cat, subcat, display in MERCHANT_DB
]


# ══════════════════════════════════════════════════════════════════════════════
# UPI VPA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# Known UPI handles → category
UPI_HANDLE_MAP = {
    # Food
    'zomato':    ('Food & Dining',      'Food Delivery',  'Zomato'),
    'swiggy':    ('Food & Dining',      'Food Delivery',  'Swiggy'),
    # Shopping
    'flipkart':  ('Shopping',           'E-Commerce',     'Flipkart'),
    'amazon':    ('Shopping',           'E-Commerce',     'Amazon'),
    'myntra':    ('Shopping',           'Fashion',        'Myntra'),
    # Travel
    'uber':      ('Travel & Transport', 'Cab',            'Uber'),
    'ola':       ('Travel & Transport', 'Cab',            'Ola'),
    'rapido':    ('Travel & Transport', 'Bike Taxi',      'Rapido'),
    'irctc':     ('Travel & Transport', 'Train',          'IRCTC'),
    # Utilities
    'jio':       ('Utilities & Bills',  'Mobile',         'Jio'),
    'airtel':    ('Utilities & Bills',  'Mobile/Broadband','Airtel'),
    'bsnl':      ('Utilities & Bills',  'Mobile',         'BSNL'),
    # Payments
    'paytm':     ('UPI Transfer',       'Wallet',         'Paytm'),
    'gpay':      ('UPI Transfer',       'UPI',            'Google Pay'),
    'phonepe':   ('UPI Transfer',       'UPI',            'PhonePe'),
    # Banks
    'sbi':       ('Bank Transfer',      'SBI',            'SBI'),
    'hdfc':      ('Bank Transfer',      'HDFC Bank',      'HDFC Bank'),
    'icici':     ('Bank Transfer',      'ICICI Bank',     'ICICI Bank'),
    'axis':      ('Bank Transfer',      'Axis Bank',      'Axis Bank'),
    'kotak':     ('Bank Transfer',      'Kotak Bank',     'Kotak Bank'),
    'ybl':       ('Bank Transfer',      'Yes Bank',       'Yes Bank (YBL)'),
    'oksbi':     ('Bank Transfer',      'SBI',            'SBI (okSBI)'),
    'okicici':   ('Bank Transfer',      'ICICI Bank',     'ICICI (okICICI)'),
    'okaxis':    ('Bank Transfer',      'Axis Bank',      'Axis (okAxis)'),
    'okhdfcbank':('Bank Transfer',      'HDFC Bank',      'HDFC (okHDFCBank)'),
    'ibl':       ('Bank Transfer',      'IndusInd Bank',  'IndusInd'),
    'paytmbank': ('Bank Transfer',      'Paytm Bank',     'Paytm Bank'),
    'airtelpaymentsbank': ('Bank Transfer', 'Airtel Bank', 'Airtel Payments Bank'),
    'fbl':       ('Bank Transfer',      'Federal Bank',   'Federal Bank'),
    'rbl':       ('Bank Transfer',      'RBL Bank',       'RBL Bank'),
    'idbi':      ('Bank Transfer',      'IDBI Bank',      'IDBI Bank'),
    'barodampay':('Bank Transfer',      'Bank of Baroda', 'Bank of Baroda'),
    'cbin':      ('Bank Transfer',      'Central Bank',   'Central Bank'),
    'cnrb':      ('Bank Transfer',      'Canara Bank',    'Canara Bank'),
    'uba':       ('Bank Transfer',      'Union Bank',     'Union Bank'),
    'utbi':      ('Bank Transfer',      'United Bank',    'United Bank'),
    'psb':       ('Bank Transfer',      'Punjab & Sind Bank','Punjab & Sind Bank'),
    'apb':       ('Bank Transfer',      'Andhra Pragathi', 'Andhra Pragathi'),
    'mahb':      ('Bank Transfer',      'Bank of Maharashtra','Bank of Maharashtra'),
    'barb':      ('Bank Transfer',      'Bank of Baroda', 'Bank of Baroda'),
    'punb':      ('Bank Transfer',      'PNB',            'Punjab National Bank'),
    'oka':       ('UPI Transfer',       'UPI',            'Udio/UPI'),
    'ptys':      ('UPI Transfer',       'Paytm',          'Paytm Merchant'),
    'pthdfc':    ('UPI Transfer',       'Paytm-HDFC',     'Paytm-HDFC'),
    'paytmqr':   ('UPI Transfer',       'Paytm QR',       'Paytm QR'),
}


def _extract_vpa(narration: str) -> Optional[str]:
    """Extract the UPI VPA (Virtual Payment Address) from narration."""
    # Typical format: UPI/TRXNID/TIME/UPI/VPA/...
    parts = re.split(r'[/\\|]', narration.strip())
    for part in parts:
        part = part.strip()
        if '@' in part and len(part) > 3:
            return part.lower()
    return None


def _vpa_handle(vpa: str) -> Optional[str]:
    """Extract the bank/service handle from VPA (the part after @)."""
    if '@' in vpa:
        return vpa.split('@', 1)[1].lower().strip()
    return None


def _vpa_username(vpa: str) -> Optional[str]:
    """Extract the username/merchant part from VPA (the part before @)."""
    if '@' in vpa:
        return vpa.split('@', 1)[0].lower().strip()
    return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def classify_transaction(
    narration: str,
    amount: float = 0.0,
    is_credit: bool = False,
    existing_category: Optional[str] = None,
) -> dict:
    """
    Classify a bank transaction into a meaningful category.

    Returns:
        dict with keys: category, subcategory, display_name, confidence
    """
    if not narration:
        return _default(is_credit)

    narration_clean = narration.strip()
    narration_lower = narration_clean.lower()

    # ── 1. Credits: detect salary / refund / interest first ──────────────────
    if is_credit:
        for pat, cat, subcat, disp in _COMPILED:
            if cat == 'Income' and pat.search(narration_lower):
                return {'category': cat, 'subcategory': subcat,
                        'display_name': disp, 'confidence': 'high'}
        # Large credit likely salary
        if amount > 10000 and any(k in narration_lower for k in
                                   ['cr', 'credit', 'salary', 'payroll', 'inf/']):
            return {'category': 'Income', 'subcategory': 'Salary',
                    'display_name': 'Salary/Income', 'confidence': 'medium'}
        if 'reversal' in narration_lower or 'refund' in narration_lower:
            return {'category': 'Income', 'subcategory': 'Refund',
                    'display_name': 'Refund/Reversal', 'confidence': 'high'}

    # ── 2. Extract VPA and match handle ──────────────────────────────────────
    vpa = _extract_vpa(narration_clean)
    if vpa:
        handle = _vpa_handle(vpa)
        username = _vpa_username(vpa)

        # Direct handle lookup
        if handle and handle in UPI_HANDLE_MAP:
            cat, subcat, disp = UPI_HANDLE_MAP[handle]
            return {'category': cat, 'subcategory': subcat,
                    'display_name': disp, 'confidence': 'high'}

        # Username + handle — try merchant DB first on the full VPA
        if username:
            for compiled_pat, cat, subcat, disp in _COMPILED:
                if compiled_pat.search(username):
                    return {'category': cat, 'subcategory': subcat,
                            'display_name': disp, 'confidence': 'high'}

    # ── 3. Full narration merchant pattern match ──────────────────────────────
    for compiled_pat, cat, subcat, disp in _COMPILED:
        if compiled_pat.search(narration_lower):
            return {'category': cat, 'subcategory': subcat,
                    'display_name': disp, 'confidence': 'high'}

    # ── 4. Keyword-based heuristics ───────────────────────────────────────────
    cat = _keyword_heuristic(narration_lower, amount, is_credit)
    if cat:
        return cat

    # ── 5. Use existing category if it's not generic ─────────────────────────
    if existing_category and existing_category.lower() not in (
        'other', 'others', 'upi transfer', 'general', ''
    ):
        return {'category': existing_category, 'subcategory': '',
                'display_name': existing_category, 'confidence': 'low'}

    # ── 6. Fallback ───────────────────────────────────────────────────────────
    # If UPI but no match, call it peer transfer
    if 'upi' in narration_lower:
        return {'category': 'UPI Transfer', 'subcategory': 'Peer Transfer',
                'display_name': 'UPI Transfer', 'confidence': 'low'}

    return _default(is_credit)


def _keyword_heuristic(text: str, amount: float, is_credit: bool) -> Optional[dict]:
    """Broad keyword catch-all when pattern matching fails."""
    rules = [
        (['food', 'restaurant', 'biryani', 'chicken', 'hotel', 'eat', 'cafe',
          'bakery', 'sweet', 'mithai', 'juice', 'chai', 'coffee', 'dabba'],
         'Food & Dining', 'Restaurant', 'Food & Dining'),
        (['grocery', 'sabzi', 'vegetable', 'fruit', 'kirana', 'fresh', 'mart',
          'store', 'supermarket'],
         'Groceries', 'Grocery', 'Groceries'),
        (['cab', 'taxi', 'auto', 'rickshaw', 'bus', 'metro', 'rail', 'train',
          'flight', 'toll', 'parking', 'petrol', 'diesel', 'fuel', 'pump'],
         'Travel & Transport', 'Transport', 'Transport'),
        (['electric', 'electricity', 'power', 'gas', 'water', 'mobile', 'recharge',
          'broadband', 'internet', 'wifi', 'dth', 'cable', 'bill', 'utility'],
         'Utilities & Bills', 'Utilities', 'Utilities & Bills'),
        (['medical', 'medicine', 'hospital', 'doctor', 'clinic', 'pharmacy',
          'health', 'lab', 'test', 'diagnostic'],
         'Healthcare', 'Medical', 'Healthcare'),
        (['school', 'college', 'university', 'course', 'fee', 'education',
          'tuition', 'coaching', 'learning'],
         'Education', 'Education', 'Education'),
        (['movie', 'cinema', 'theatre', 'entertainment', 'game', 'sport',
          'music', 'concert', 'event', 'show', 'subscription'],
         'Entertainment', 'Entertainment', 'Entertainment'),
        (['rent', 'maintenance', 'society', 'housing', 'flat', 'apartment',
          'pg\b', 'hostel'],
         'Housing & Rent', 'Housing', 'Housing & Rent'),
        (['invest', 'mutual fund', 'sip\b', 'stock', 'share', 'demat',
          'ppf', 'nps', 'lic', 'insurance', 'fd\b', 'rd\b'],
         'Investments', 'Investment', 'Investments'),
        (['tax', 'gst', 'tds', 'income tax', 'challan', 'government', 'govt'],
         'Taxes & Government', 'Tax', 'Taxes & Government'),
        (['salary', 'payroll', 'stipend', 'wage'],
         'Income', 'Salary', 'Salary'),
    ]
    for keywords, cat, subcat, disp in rules:
        pat = '|'.join(r'\b' + k + r'\b' for k in keywords)
        if re.search(pat, text, re.IGNORECASE):
            return {'category': cat, 'subcategory': subcat,
                    'display_name': disp, 'confidence': 'medium'}
    return None


def _default(is_credit: bool) -> dict:
    if is_credit:
        return {'category': 'Income', 'subcategory': 'Other Credit',
                'display_name': 'Income/Credit', 'confidence': 'low'}
    return {'category': 'Other', 'subcategory': 'Miscellaneous',
            'display_name': 'Other', 'confidence': 'low'}


# ══════════════════════════════════════════════════════════════════════════════
# BATCH ENRICHMENT HELPER
# ══════════════════════════════════════════════════════════════════════════════

def enrich_transactions(transactions: list) -> list:
    """
    Re-classify a list of transaction dicts in place.
    Expected keys: narration/description, debit, credit, category (optional)
    Returns the list with updated category fields.
    """
    enriched = []
    for txn in transactions:
        narration = (txn.get('narration') or txn.get('description')
                     or txn.get('particulars') or '')
        debit  = float(txn.get('debit')  or 0)
        credit = float(txn.get('credit') or 0)
        is_credit = credit > 0 and debit == 0
        amount = credit if is_credit else debit
        old_cat = txn.get('category', '')

        result = classify_transaction(narration, amount, is_credit, old_cat)

        txn_copy = dict(txn)
        txn_copy['category']    = result['category']
        txn_copy['subcategory'] = result.get('subcategory', '')
        txn_copy['merchant']    = result.get('display_name', '')
        enriched.append(txn_copy)

    return enriched


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY ICON MAP (for UI badges)
# ══════════════════════════════════════════════════════════════════════════════
CATEGORY_ICONS = {
    'Food & Dining':          '🍽️',
    'Groceries':              '🛒',
    'Travel & Transport':     '🚗',
    'Utilities & Bills':      '⚡',
    'Shopping':               '🛍️',
    'Healthcare':             '🏥',
    'Education':              '📚',
    'Entertainment':          '🎬',
    'Investments':            '📈',
    'Insurance':              '🛡️',
    'Loan/EMI':               '🏦',
    'Housing & Rent':         '🏠',
    'Bank Transfer':          '🔄',
    'UPI Transfer':           '📱',
    'Income':                 '💰',
    'Taxes & Government':     '🏛️',
    'Cash & ATM':             '💵',
    'Charity & Donations':    '❤️',
    'Other':                  '📂',
}

CATEGORY_COLORS = {
    'Food & Dining':          '#f59e0b',
    'Groceries':              '#10b981',
    'Travel & Transport':     '#3b82f6',
    'Utilities & Bills':      '#6366f1',
    'Shopping':               '#a855f7',
    'Healthcare':             '#f43f5e',
    'Education':              '#0ea5e9',
    'Entertainment':          '#ec4899',
    'Investments':            '#14b8a6',
    'Insurance':              '#64748b',
    'Loan/EMI':               '#ef4444',
    'Housing & Rent':         '#f97316',
    'Bank Transfer':          '#94a3b8',
    'UPI Transfer':           '#8b5cf6',
    'Income':                 '#22c55e',
    'Taxes & Government':     '#6b7280',
    'Cash & ATM':             '#d97706',
    'Charity & Donations':    '#ec4899',
    'Other':                  '#94a3b8',
}


if __name__ == '__main__':
    # Quick smoke test
    test_cases = [
        ('UPI/645768964223/15:02:04/UPI/8390158877@yapl/UPI', 40, False),
        ('UPI/609203076028/20:13:50/UPI/zaidrahman342-1@oka', 70, False),
        ('UPI/646596265799/23:50:03/UPI/37320100034985@barb', 1, False),
        ('UPI/646867880789/15:30:51/UPI/gpay-11256937782@ok', 84, False),
        ('Salary Credit HDFC PAYROLL', 45000, True),
        ('ZOMATO ORDER 9876', 350, False),
        ('NETFLIX SUBSCRIPTION', 649, False),
        ('IRCTC TRAIN TICKET', 1200, False),
        ('HPCL PETROL PUMP', 2000, False),
        ('MSEDCL ELECTRICITY BILL', 1800, False),
        ('BigBasket grocery order', 1500, False),
        ('Amazon Prime subscription', 1499, False),
        ('Axis Bank ATM CASH WITHDRAWAL', 5000, False),
    ]
    print(f"{'Narration':<55} {'Category':<25} {'Subcategory':<22} {'Display'}")
    print('─' * 130)
    for narr, amt, cred in test_cases:
        r = classify_transaction(narr, amt, cred)
        icon = CATEGORY_ICONS.get(r['category'], '📂')
        print(f"{narr[:54]:<55} {icon} {r['category']:<23} {r['subcategory']:<22} {r['display_name']}")