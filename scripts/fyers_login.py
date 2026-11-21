#!/usr/bin/env python3
"""Interactive CLI login utility for FYERS API v3.

Usage:
    python scripts/fyers_login.py
"""

import sys
import webbrowser

from nifty_engine.data.fyers.auth import FyersAuthProvider


def main():
    print("=" * 60)
    print("  NIFTY Options Trading Intelligence - FYERS Login Wizard")
    print("=" * 60)

    auth = FyersAuthProvider()

    if not auth.app_id or not auth.secret_key:
        print("\n[ERROR] FYERS_APP_ID or FYERS_SECRET_KEY is missing in your .env file!")
        print("Please edit .env and enter your FYERS developer credentials.\n")
        sys.exit(1)

    # Check existing token
    existing_token = auth.load_cached_token()
    if existing_token:
        print("\n[SUCCESS] A valid FYERS access token is already cached on this machine.")
        print(f"Token file: {auth.token_file}")
        re_login = input("\nDo you want to re-authenticate and generate a new token? (y/N): ").strip().lower()
        if re_login != "y":
            print("Exiting without changes.")
            return

    # Generate login URL
    login_url = auth.generate_login_url()
    print("\n1. Opening FYERS official authorization portal in your browser:")
    print(f"\n   {login_url}\n")

    try:
        webbrowser.open(login_url)
    except Exception:
        pass

    print("2. Log in using your FYERS credentials and 2FA.")
    print("3. Upon authorization, your browser will redirect to a page containing the 'auth_code'.")
    print("   Example URL: https://trade.fyers.in/api-login/redirect-uri/index.html?s=ok&code=YOUR_CODE_HERE\n")

    auth_code = input("Paste the 'code' parameter value here: ").strip()

    if not auth_code:
        print("\n[ERROR] No code entered. Aborting.")
        sys.exit(1)

    print("\nExchanging auth code for daily access token...")
    try:
        _ = auth.exchange_code_for_token(auth_code)
        print("\n[SUCCESS] Authentication complete!")
        print(f"Daily token cached at: {auth.token_file} (Mode 0600)")
        print("Valid until 6:30 AM IST tomorrow.")
    except Exception as e:
        print(f"\n[FAILED] Error during token exchange: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
