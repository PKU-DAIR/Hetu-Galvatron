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
  // Initialize language from localStorage or default to Chinese
  const [language, setLanguage] = useState(() => {
    const savedLanguage = localStorage.getItem('galvatron-language');
    return savedLanguage || LANGUAGES.ZH;
  });

  // Save language preference to localStorage
  useEffect(() => {
    localStorage.setItem('galvatron-language', language);
  }, [language]);

  // Toggle between languages
  const toggleLanguage = () => {
    setLanguage(prevLang => 
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