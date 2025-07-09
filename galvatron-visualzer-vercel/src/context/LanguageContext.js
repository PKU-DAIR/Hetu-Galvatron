import React, { createContext, useState, useContext, useEffect } from 'react';

// Language context to manage language preferences
const LanguageContext = createContext();

// Available languages
export const LANGUAGES = {
  ZH: 'zh',
  EN: 'en'
};

// Language provider component
export const LanguageProvider = ({ children }) => {
    // Init language to a default value. This will be used for SSR and initial rendering 
    const [language, setLanguage] = useState(LANGUAGES.ZH);
    // Effect to load and apply language from localStorage ONLY on the client-side, after hydration.
    useEffect(() => {
        // Check if running on the client
        if (typeof window !== "undefined") {
            const savedLanguage = localStorage.getItem("galvatron-language");
            if (savedLanguage && savedLanguage !== language) {
                // Check if it's a valid language from your LANGUAGES object
                if (Object.values(LANGUAGES).includes(savedLanguage)) {
                    setLanguage(savedLanguage);
                } else {
                    // Optional: handle invalid value in localStorage, e.g., reset to default
                    localStorage.setItem("galvatron-language", LANGUAGES.ZH);
                }
            }
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []); // Empty dependency array ensures this runs only once on mount (after hydration)

    // Save language preference to localStorage
    useEffect(() => {
        // Only access localStorage on the client-side
        if (typeof window !== "undefined") {
            localStorage.setItem("galvatron-language", language);
        }
    }, [language]);

    // Toggle between languages
    const toggleLanguage = () => {
        setLanguage((prevLang) =>
            prevLang === LANGUAGES.ZH ? LANGUAGES.EN : LANGUAGES.ZH
        );
    };

    return (
        <LanguageContext.Provider value={{ language, toggleLanguage }}>
            {children}
        </LanguageContext.Provider>
    );
};

// Custom hook to use language context
export const useLanguage = () => useContext(LanguageContext);

export default LanguageContext;