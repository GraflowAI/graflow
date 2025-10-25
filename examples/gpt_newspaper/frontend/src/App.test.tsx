import { render, screen } from "@testing-library/react";
import App from "./App";

vi.mock("./hooks/useNewspaper", () => ({
  useNewspaper: () => ({
    history: [],
    latest: null,
    loading: false,
    error: null,
    generate: vi.fn(),
    refreshHistory: vi.fn()
  })
}));

describe("App", () => {
  it("renders headline and form instructions", () => {
    render(<App />);

    expect(screen.getByText(/GPT Newspaper/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Queries/i)).toBeInTheDocument();
  });
});
