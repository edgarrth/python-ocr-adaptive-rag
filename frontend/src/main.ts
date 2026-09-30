import { bootstrapApplication } from '@angular/platform-browser';
import { provideHttpClient } from '@angular/common/http';
import { AppComponent } from './app/app';

bootstrapApplication(AppComponent, {
  providers: [provideHttpClient()]
}).catch((error) => console.error(error));
