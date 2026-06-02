//! SEC-EDGAR submissions parser (the Python `SECEdgarPipeline` / `SECFilingTracker`).
//!
//! Parses `data.sec.gov/submissions/CIK##########.json` into a flat list of
//! recent filings. Form 4 counts feed the insider-transaction signal. The
//! parser is pure and tested offline; the live fetch needs the polite
//! User-Agent enforced by [`crate::http::HttpClient`].

use serde::Deserialize;

use sovereign_core::error::{Result, SovereignError};

#[derive(Debug, Deserialize)]
struct Submissions {
    filings: Filings,
}

#[derive(Debug, Deserialize)]
struct Filings {
    recent: Recent,
}

#[derive(Debug, Deserialize)]
struct Recent {
    #[serde(rename = "accessionNumber", default)]
    accession_number: Vec<String>,
    #[serde(default)]
    form: Vec<String>,
    #[serde(rename = "filingDate", default)]
    filing_date: Vec<String>,
}

/// A single filing.
#[derive(Debug, Clone, PartialEq)]
pub struct Filing {
    pub form: String,
    pub date: String,
    pub accession: String,
}

/// Parse the recent-filings block into a flat list (the three parallel arrays
/// are zipped by index, truncated to the shortest).
pub fn parse_recent_filings(body: &str) -> Result<Vec<Filing>> {
    let sub: Submissions = serde_json::from_str(body).map_err(|e| SovereignError::Serde {
        context: "sec_edgar".into(),
        source: e,
    })?;
    let r = sub.filings.recent;
    let n = r
        .form
        .len()
        .min(r.filing_date.len())
        .min(r.accession_number.len());
    Ok((0..n)
        .map(|i| Filing {
            form: r.form[i].clone(),
            date: r.filing_date[i].clone(),
            accession: r.accession_number[i].clone(),
        })
        .collect())
}

/// Count Form 4 (insider transaction) filings — the bullishness proxy.
pub fn count_form4(filings: &[Filing]) -> usize {
    filings.iter().filter(|f| f.form == "4").count()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "cik": "320193",
        "filings": {
            "recent": {
                "accessionNumber": ["0001-26-000001", "0001-26-000002", "0001-26-000003"],
                "form": ["10-Q", "4", "4"],
                "filingDate": ["2026-05-01", "2026-05-10", "2026-05-12"]
            }
        }
    }"#;

    #[test]
    fn parses_filings_and_counts_form4() {
        let filings = parse_recent_filings(SAMPLE).unwrap();
        assert_eq!(filings.len(), 3);
        assert_eq!(filings[0].form, "10-Q");
        assert_eq!(count_form4(&filings), 2);
    }

    #[test]
    fn missing_fields_default_to_empty() {
        let filings = parse_recent_filings(r#"{"filings":{"recent":{}}}"#).unwrap();
        assert!(filings.is_empty());
    }
}
