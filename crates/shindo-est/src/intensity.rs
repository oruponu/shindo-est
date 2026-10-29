//! Japan Meteorological Agency (JMA) seismic intensity classes.

use std::fmt;

/// Lowest instrumental intensity of each class above [`IntensityClass::Zero`],
/// in ascending order.
const LOWER_BOUNDS: [(f64, IntensityClass); 9] = [
    (0.5, IntensityClass::One),
    (1.5, IntensityClass::Two),
    (2.5, IntensityClass::Three),
    (3.5, IntensityClass::Four),
    (4.5, IntensityClass::FiveLower),
    (5.0, IntensityClass::FiveUpper),
    (5.5, IntensityClass::SixLower),
    (6.0, IntensityClass::SixUpper),
    (6.5, IntensityClass::Seven),
];

/// A class of the JMA seismic intensity scale. The discriminants 0 to 9 are
/// stable for the C ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum IntensityClass {
    /// Intensity 0.
    Zero = 0,
    /// Intensity 1.
    One = 1,
    /// Intensity 2.
    Two = 2,
    /// Intensity 3.
    Three = 3,
    /// Intensity 4.
    Four = 4,
    /// Intensity 5 lower, labeled "5-".
    FiveLower = 5,
    /// Intensity 5 upper, labeled "5+".
    FiveUpper = 6,
    /// Intensity 6 lower, labeled "6-".
    SixLower = 7,
    /// Intensity 6 upper, labeled "6+".
    SixUpper = 8,
    /// Intensity 7.
    Seven = 9,
}

impl IntensityClass {
    /// Returns the class of an instrumental intensity, or `None` for NaN.
    ///
    /// Each class includes its lower bound, and the value is compared as is,
    /// without rounding to one decimal place.
    #[must_use]
    pub fn from_instrumental(value: f64) -> Option<Self> {
        if value.is_nan() {
            return None;
        }
        for &(bound, class) in LOWER_BOUNDS.iter().rev() {
            if value >= bound {
                return Some(class);
            }
        }
        Some(Self::Zero)
    }

    /// Returns the short ASCII label: "0" to "4", "5-", "5+", "6-", "6+", or "7".
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Zero => "0",
            Self::One => "1",
            Self::Two => "2",
            Self::Three => "3",
            Self::Four => "4",
            Self::FiveLower => "5-",
            Self::FiveUpper => "5+",
            Self::SixLower => "6-",
            Self::SixUpper => "6+",
            Self::Seven => "7",
        }
    }
}

impl fmt::Display for IntensityClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [IntensityClass; 10] = [
        IntensityClass::Zero,
        IntensityClass::One,
        IntensityClass::Two,
        IntensityClass::Three,
        IntensityClass::Four,
        IntensityClass::FiveLower,
        IntensityClass::FiveUpper,
        IntensityClass::SixLower,
        IntensityClass::SixUpper,
        IntensityClass::Seven,
    ];

    #[test]
    fn boundaries_belong_to_the_upper_class() {
        // (bound, class just below it, class from it), written out apart from
        // LOWER_BOUNDS so that a moved bound fails here.
        let bounds: [(f64, IntensityClass, IntensityClass); 9] = [
            (0.5, IntensityClass::Zero, IntensityClass::One),
            (1.5, IntensityClass::One, IntensityClass::Two),
            (2.5, IntensityClass::Two, IntensityClass::Three),
            (3.5, IntensityClass::Three, IntensityClass::Four),
            (4.5, IntensityClass::Four, IntensityClass::FiveLower),
            (5.0, IntensityClass::FiveLower, IntensityClass::FiveUpper),
            (5.5, IntensityClass::FiveUpper, IntensityClass::SixLower),
            (6.0, IntensityClass::SixLower, IntensityClass::SixUpper),
            (6.5, IntensityClass::SixUpper, IntensityClass::Seven),
        ];

        for (bound, below, from) in bounds {
            let just_below = bound.next_down();
            let just_above = bound.next_up();
            assert_eq!(
                IntensityClass::from_instrumental(just_below),
                Some(below),
                "{just_below}"
            );
            assert_eq!(
                IntensityClass::from_instrumental(bound),
                Some(from),
                "{bound}"
            );
            assert_eq!(
                IntensityClass::from_instrumental(just_above),
                Some(from),
                "{just_above}"
            );
        }
    }

    #[test]
    fn values_are_not_rounded() {
        let cases = [
            (-1.0, IntensityClass::Zero),
            (0.0, IntensityClass::Zero),
            (-0.0, IntensityClass::Zero),
            (1.49, IntensityClass::One),
            (4.95, IntensityClass::FiveLower),
            (9.0, IntensityClass::Seven),
        ];

        for (value, expected) in cases {
            assert_eq!(
                IntensityClass::from_instrumental(value),
                Some(expected),
                "{value}"
            );
        }
    }

    #[test]
    fn infinities_take_the_end_classes() {
        assert_eq!(
            IntensityClass::from_instrumental(f64::NEG_INFINITY),
            Some(IntensityClass::Zero)
        );
        assert_eq!(
            IntensityClass::from_instrumental(f64::INFINITY),
            Some(IntensityClass::Seven)
        );
    }

    #[test]
    fn nan_has_no_class() {
        let nans = [f64::NAN, -f64::NAN, f64::from_bits(0x7ff8_0000_0000_0001)];

        for nan in nans {
            assert_eq!(
                IntensityClass::from_instrumental(nan),
                None,
                "{:#x}",
                nan.to_bits()
            );
        }
    }

    #[test]
    fn labels() {
        let labels: Vec<&str> = ALL.into_iter().map(IntensityClass::label).collect();

        assert_eq!(
            labels,
            ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
        );
    }

    #[test]
    fn display_writes_the_label() {
        for class in ALL {
            assert_eq!(class.to_string(), class.label(), "{class:?}");
        }
    }

    #[test]
    fn classes_are_ordered_by_strength() {
        for pair in ALL.windows(2) {
            assert!(pair[0] < pair[1], "{pair:?}");
        }
    }

    #[test]
    fn discriminants_are_0_to_9() {
        let discriminants: Vec<i32> = ALL.into_iter().map(|class| class as i32).collect();
        let expected: Vec<i32> = (0..10).collect();

        assert_eq!(discriminants, expected);
    }
}
