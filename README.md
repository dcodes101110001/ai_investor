# Build Your Own A.I. Investor 2024
* Simple Python/Machine Learning/Value Investing, with a stock picking A.I. to tie it all together.
* Build Your Own A.I. Investor materials on Gumroad.
* ai-investor.com
* https://aiinvestor.gumroad.com/l/BuildYourOwnAIInvestor

## Who is this for?
* People who follow the book/videos
* Intended to be simple enough for total beginners who have never touched Python before
* Basic Machine Learning for assisting stock picking

Stock fundamental and price data from https://simfin.com/ is required (their free data will do to run things). No affiliation (personally I think their product is good).

## Automated Data Download

This repository includes a GitHub Actions workflow that automatically downloads SimFin data daily at 2 AM UTC. To enable this:

1. Get a free API key from [SimFin.com](https://simfin.com/)
2. Add it as a repository secret named `SIMFIN_API_KEY`
3. See [SIMFIN_SETUP.md](SIMFIN_SETUP.md) for detailed setup instructions

**For Public Forks:** If you've forked this repository, you'll need to set up authentication for the workflow to push data updates. 

- **Quick Start**: See [QUICK_SETUP_PAT.md](QUICK_SETUP_PAT.md) for a 5-minute setup guide
- **Detailed Guide**: See [WORKFLOW_AUTHENTICATION_SETUP.md](WORKFLOW_AUTHENTICATION_SETUP.md) for comprehensive instructions
- **Inline Help**: The workflow file itself contains detailed setup instructions - see `.github/workflows/download-simfin-data.yml`

The data is saved to the `stock_data/` directory and refreshed daily.

## Disclaimer
The author does not make any guarantee or other promise as to any results that may be obtained from using the content. You should never make any investment decision without first consulting with your financial advisor and conducting your research and due diligence. To the maximum extent permitted by law, the author disclaims all liability in the event any information, commentary, analysis, opinions, advice and/or recommendations contained in this book prove to be inaccurate, incomplete or unreliable or result in any investment or other losses.
